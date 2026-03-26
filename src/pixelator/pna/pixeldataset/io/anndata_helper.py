"""Helper functions for materializing AnnData objects from PXL files.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import warnings
from functools import cache
from typing import Literal

import polars as pl
from anndata import AnnData, ImplicitModificationWarning
from anndata import concat as anndata_concat

from pixelator.common.statistics import clr_transformation, log1p_transformation
from pixelator.pna.pixeldataset.utils import update_metrics_anndata
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set

from .pixel_data_viewer import PixelDataViewer
from .query_builder import QueryBuilder


class AnnDataHelper:
    """Helper class to deal with materializing the AnnnData object from the pxl file."""

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
        with self._view as connection:
            adatas: list[AnnData] = []
            for sample_name in self._view.sample_names():
                adata = self._read_adata_from_sample(
                    connection=connection, sample=sample_name
                )
                adatas.append(adata)

        if not adatas:
            return AnnData()

        concatenated = anndata_concat(adatas, join=self._adata_join_strategy)
        concatenated.var = adatas[0].var
        concatenated.uns = adatas[0].uns
        update_metrics_anndata(concatenated, inplace=True)
        return concatenated

    def _read_adata_from_sample(
        self,
        *,
        connection,
        sample: str,
    ) -> AnnData:
        qb = QueryBuilder()
        db_name = self._view.normalized_sample_db_name(sample)

        # Read full AnnData contents (components/markers are filtered in-memory later).
        X = self._view.execute_eager(
            connection, qb.adata_X_query(db_name, None)
        ).to_pandas()
        var = self._view.execute_eager(
            connection, qb.adata_var_query(db_name, None)
        ).to_pandas()
        obs = self._view.execute_eager(
            connection, qb.adata_obs_query(db_name, None)
        ).to_pandas()

        maybe_uns = connection.sql(qb.adata_uns_query(db_name).sql).fetchone()
        uns = json.loads(maybe_uns[0]) if maybe_uns else None

        tables = self._view.execute_eager(
            connection, qb.adata_obsm_table_names_query(db_name)
        )
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
                self._view.execute_eager(
                    connection,
                    qb.adata_obsm_query(db_name, table, None),
                )
                .to_pandas()
                .set_index("index")
                .rename_axis(index={"index": "component"})
            )
            for table in obsm_tables
        }

        adata = AnnData(
            X=X.set_index("index").rename_axis(index={"index": "component"}),
            var=var.set_index("index").rename_axis(index={"index": "marker_id"}),
            obs=obs.set_index("index").rename_axis(index={"index": "component"}),
            uns=uns,
            obsm=obsm,
        )
        adata.obs["sample"] = sample
        return adata

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
