"""Helper functions for materializing AnnData objects from PXL files.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Literal, Mapping

import pandas as pd
import polars as pl
from anndata import AnnData, ImplicitModificationWarning
from anndata import concat as anndata_concat

from pixelator.common.statistics import clr_transformation, log1p_transformation
from pixelator.common.utils import logger
from pixelator.pna.config.panel import ORIGINAL_HASH_COUNTS_PREFIX
from pixelator.pna.config.panel.tables import has_panel_tables
from pixelator.pna.pixeldataset.utils import update_metrics_anndata
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set

from .pixel_data_viewer import PixelDataViewer, PixelDataViewerSession
from .query_builder import QueryBuilder

_MAX_REPORTED_MARKERS = 10


class AnnDataHelper:
    """Helper class to deal with materializing the AnnData object from the pxl file."""

    def __init__(
        self,
        view: PixelDataViewer,
        components: str | list[str] | set[str] | None = None,
        markers: str | list[str] | set[str] | None = None,
        adata_join_strategy: Literal["inner", "outer"] = "inner",
    ):
        """Create a new instance of AnnDataHelper."""
        self._view = view
        self._components = normalize_input_to_set(components)
        self._markers = normalize_input_to_set(markers)
        self._adata_join_strategy = adata_join_strategy

    def _read_all_samples(self) -> AnnData:
        """Read and concatenate AnnData from all samples in the current view.

        Returns:
            A concatenated AnnData object. Returns an empty AnnData when no
            samples are available.
        """
        with self._view.open() as session:
            adatas: list[AnnData] = []
            for sample_name in self._view.sample_names():
                adata = self._read_adata_from_sample(
                    session=session, sample=sample_name
                )
                adatas.append(adata)

        if not adatas:
            return AnnData()

        # fill_value only applies to the outer join, where it counts the markers a
        # sample lacks as 0 (rather than NaN, which the metrics would count as present).
        concatenated = anndata_concat(
            adatas, join=self._adata_join_strategy, fill_value=0
        )
        # The first sample wins for markers that several samples have. Reindex to the
        # joined markers, as the first sample may have markers the join dropped.
        var = pd.concat([adata.var for adata in adatas])
        var = var[~var.index.duplicated(keep="first")]
        concatenated.var = var.loc[concatenated.var_names]
        if self._adata_join_strategy == "inner":
            self._warn_about_dropped_markers(adatas)
        concatenated.uns = adatas[0].uns
        update_metrics_anndata(concatenated, inplace=True)
        return concatenated

    @staticmethod
    def _warn_about_dropped_markers(adatas: list[AnnData]) -> None:
        """Warn about the markers an inner join drops, as not all samples have them."""
        all_markers = set().union(*(adata.var_names for adata in adatas))
        shared_markers = set(adatas[0].var_names).intersection(
            *(adata.var_names for adata in adatas[1:])
        )
        dropped = sorted(all_markers - shared_markers)
        if not dropped:
            return
        shown = ", ".join(dropped[:_MAX_REPORTED_MARKERS])
        more = len(dropped) - _MAX_REPORTED_MARKERS
        logger.warning(
            "Markers that are not in all samples are dropped (%d): %s%s",
            len(dropped),
            shown,
            f" and {more} more" if more > 0 else "",
        )

    def _read_adata_from_sample(
        self,
        *,
        session: PixelDataViewerSession,
        sample: str,
    ) -> AnnData:
        qb = QueryBuilder()
        db_name = self._view.normalized_sample_db_name(sample)

        # How this sample's marker ids are renamed to line up with the other samples.
        renames = self._view.marker_renames().get(sample, {})
        hash_count_renames = {
            f"{ORIGINAL_HASH_COUNTS_PREFIX}{old}": f"{ORIGINAL_HASH_COUNTS_PREFIX}{new}"
            for old, new in renames.items()
        }

        # Read full AnnData contents (components/markers are filtered in-memory later).
        X = session.execute_eager(
            qb.adata_X_query(
                db_name,
                None,
                column_renames=self._renames_of_stored_columns(
                    session, qb, f"{db_name}.__adata__X", renames
                ),
            )
        ).to_pandas()
        var = session.execute_eager(
            qb.adata_var_query(db_name, None, marker_renames=renames)
        ).to_pandas()
        obs = session.execute_eager(
            qb.adata_obs_query(
                db_name,
                None,
                column_renames=self._renames_of_stored_columns(
                    session, qb, f"{db_name}.__adata__obs", hash_count_renames
                ),
            )
        ).to_pandas()

        uns_df = session.execute_eager(qb.adata_uns_query(db_name))
        uns = json.loads(uns_df.row(0)[0]) if not uns_df.is_empty() else None

        tables = session.execute_eager(qb.adata_obsm_table_names_query(db_name))
        obsm_tables = (
            tables.lazy()
            .filter(
                (pl.col("name").str.starts_with("__adata__obsm"))
                & (pl.col("database") == db_name)
            )
            .select(
                pl.concat_str(
                    [pl.col("database"), pl.col("schema"), pl.col("name")],
                    separator=".",
                ).alias("name")
            )
            .collect()
            .get_column("name")
            .to_list()
        )

        obsm = {
            table.split("__adata__obsm_")[1]: (
                session.execute_eager(
                    qb.adata_obsm_query(
                        db_name,
                        table,
                        None,
                        column_renames=self._renames_of_stored_columns(
                            session, qb, table, renames
                        ),
                    ),
                )
                .to_pandas()
                .set_index("index")
                .rename_axis(index={"index": "component"})
            )
            for table in obsm_tables
        }

        var = var.set_index("index").rename_axis(index={"index": "marker_id"})
        var = self._with_panel_info(session, qb, db_name, var)

        adata = AnnData(
            X=X.set_index("index").rename_axis(index={"index": "component"}),
            var=var,
            obs=obs.set_index("index").rename_axis(index={"index": "component"}),
            uns=uns,
            obsm=obsm,
        )
        adata.obs["sample"] = sample
        return adata

    @staticmethod
    def _renames_of_stored_columns(
        session: PixelDataViewerSession,
        qb: QueryBuilder,
        table: str,
        renames: Mapping[str, str],
    ) -> dict[str, str]:
        """Return the renames that apply to columns ``table`` has.

        A rename of a column the table lacks is dropped, for example a hashing
        clone in a table written after sample calling.
        """
        if not renames:
            return {}
        columns = set(
            session.execute_eager(qb.table_columns_query(table))[
                "column_name"
            ].to_list()
        )
        return {old: new for old, new in renames.items() if old in columns}

    @staticmethod
    def _has_panel_tables(
        session: PixelDataViewerSession, qb: QueryBuilder, db_name: str
    ) -> bool:
        """Return whether the file of ``db_name`` stores its panel in tables."""
        tables = session.execute_eager(qb.adata_obsm_table_names_query(db_name))
        return has_panel_tables(
            tables.filter(pl.col("database") == db_name).get_column("name").to_list()
        )

    def _with_panel_info(
        self,
        session: PixelDataViewerSession,
        qb: QueryBuilder,
        db_name: str,
        var: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return ``var`` with the panel columns of its markers.

        When the view aligned the panels of its files, the columns come from the
        aligned panel rows. This makes ``var`` the same whichever sample is read
        first, as the concatenated AnnData keeps the ``var`` of the first sample.
        Otherwise they come from the panel tables of the file, if it has them.
        Older files keep the panel columns on ``var`` and are left as they are.
        """
        aligned_rows = self._view.aligned_panel_rows()
        if aligned_rows:
            panel_markers = pd.DataFrame.from_dict(aligned_rows, orient="index").drop(
                columns="marker_id"
            )
        elif self._has_panel_tables(session, qb, db_name):
            panel_markers = (
                session.execute_eager(qb.panel_markers_query(db_name))
                .to_pandas()
                .set_index("marker_id")
                .drop(columns=["row_nr", "source_id"])
            )
        else:
            return var
        # panel columns already on var (older files) are replaced
        var = var.drop(columns=var.columns.intersection(panel_markers.columns))
        return var.join(panel_markers.reindex(var.index), how="left")

    def _apply_transformations(
        self,
        adata: AnnData,
        *,
        add_log1p_transform: bool,
        add_clr_transform: bool,
    ) -> AnnData:
        if [add_clr_transform, add_log1p_transform].count(True) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
                counts_df = adata.to_df()
                if add_clr_transform:
                    counts_df_clr = clr_transformation(df=counts_df, axis=1)
                    counts_df_clr.index = counts_df_clr.index.astype(str)
                    adata.obsm["clr"] = counts_df_clr

                if add_log1p_transform:
                    counts_df_log1p = log1p_transformation(df=counts_df)
                    counts_df_log1p.index = counts_df_log1p.index.astype(str)
                    adata.obsm["log1p"] = counts_df_log1p
        return adata

    @cache
    def _read_adata_cached(
        self,
        *,
        add_log1p_transform: bool = True,
        add_clr_transform: bool = True,
    ) -> AnnData:
        """Materialize the AnnData object once for the given options.

        Note on caching + mutability:
        `AnnData` is mutable (callers can add/remove layers and other fields).
        Therefore, this *cached* method returns an internal "canonical" instance
        that must never be returned directly to callers.
        """
        adata = self._read_all_samples()

        if self._components:
            adata = adata[normalize_input_to_list(self._components), :]
        if self._markers:
            adata = adata[:, normalize_input_to_list(self._markers)]

        adata = self._apply_transformations(
            adata,
            add_log1p_transform=add_log1p_transform,
            add_clr_transform=add_clr_transform,
        )
        # Return a fully-materialized canonical object for this cache key.
        return adata.copy()

    def read_adata(
        self,
        *,
        add_log1p_transform: bool = True,
        add_clr_transform: bool = True,
    ) -> AnnData:
        """Return a filtered/transformed AnnData instance.

        The returned object is always a defensive copy of the cached canonical
        value, so caller mutations never leak back into the cache.
        """
        return self._read_adata_cached(
            add_log1p_transform=add_log1p_transform,
            add_clr_transform=add_clr_transform,
        ).copy()
