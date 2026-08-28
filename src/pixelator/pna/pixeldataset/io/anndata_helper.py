"""Helper functions for materializing AnnData objects from PXL files.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from functools import cache
from typing import Literal

import polars as pl
import semver
from anndata import AnnData, ImplicitModificationWarning
from anndata import concat as anndata_concat

from pixelator.common.statistics import clr_transformation, log1p_transformation
from pixelator.common.utils import logger
from pixelator.pna.config.panel import (
    PartialPNAAntibodyPanel,
    PNAAntibodyPanelCombination,
    PNAAntibodyPanelDiff,
)
from pixelator.pna.pixeldataset.utils import update_metrics_anndata
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set
from pixelator.pna.utils.sample_calling_uns import sample_calling_hashing_collapsed

from .pixel_data_viewer import PixelDataViewer, PixelDataViewerSession
from .query_builder import QueryBuilder


def remap_marker_id_columns(
    df: pl.DataFrame,
    remaps: dict[str, dict[str, str]],
    columns: tuple[str, ...] = ("marker_1", "marker_2"),
) -> pl.DataFrame:
    """Rename marker id columns using per-sample old→new maps from a panel patch bump."""
    if not remaps or not any(remaps.values()):
        return df
    rows: list[dict[str, str]] = [
        {"sample": sample, "old_id": old, "new_id": new}
        for sample, mapping in remaps.items()
        for old, new in mapping.items()
    ]
    if not rows:
        return df
    map_df = pl.DataFrame(rows)
    upgraded = df
    join_on_sample = "sample" in upgraded.columns
    for column in columns:
        if column not in upgraded.columns:
            continue
        if join_on_sample:
            upgraded = (
                upgraded.join(
                    map_df,
                    left_on=["sample", column],
                    right_on=["sample", "old_id"],
                    how="left",
                )
                .with_columns(pl.coalesce("new_id", column).alias(column))
                .drop("new_id")
            )
        else:
            per_id = map_df.select("old_id", "new_id").unique(
                subset=["old_id"], keep="first"
            )
            upgraded = (
                upgraded.join(per_id, left_on=column, right_on="old_id", how="left")
                .with_columns(pl.coalesce("new_id", column).alias(column))
                .drop("new_id")
            )
    return upgraded


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
        self._marker_id_renames_by_sample: dict[str, dict[str, str]] = {}

    def _read_all_samples(self) -> AnnData:
        """Read and concatenate AnnData from all samples in the current view.

        Returns:
            A concatenated AnnData object. Returns an empty AnnData when no
            samples are available.

        Raises:
            ValueError: If sample-level ``var`` tables cannot be aligned during
                concatenation.
        """
        sample_names: list[str] = []
        adatas: list[AnnData] = []
        with self._view.open() as session:
            for sample_name in self._view.sample_names():
                adata = self._read_adata_from_sample(
                    session=session, sample=sample_name
                )
                sample_names.append(sample_name)
                adatas.append(adata)

        self._try_bump_adata_panel_version(adatas, sample_names)

        if not adatas:
            return AnnData()

        concatenated = anndata_concat(adatas, join=self._adata_join_strategy)
        try:
            concatenated.var = adatas[0].var
        except ValueError as err:
            raise ValueError(
                "Failed to concatenate AnnData var - check that all samples have the same set of markers."
            ) from err
        concatenated.uns = adatas[0].uns
        update_metrics_anndata(concatenated, inplace=True)
        return concatenated

    def marker_id_renames_by_sample(self) -> dict[str, dict[str, str]]:
        """Return old→new marker_id maps applied by panel patch bumps, per sample.

        Triggers the same AnnData materialization path as :meth:`read_adata` so
        the maps match the upgraded ``var`` index.
        """
        self._read_adata_cached(add_log1p_transform=False, add_clr_transform=False)
        return dict(self._marker_id_renames_by_sample)

    def apply_marker_id_renames(
        self,
        df: pl.DataFrame,
        columns: tuple[str, ...] = ("marker_1", "marker_2"),
    ) -> pl.DataFrame:
        """Rename marker id columns to match a panel patch bump of ``var``."""
        return remap_marker_id_columns(df, self.marker_id_renames_by_sample(), columns)

    def marker_ids_for_on_disk_query(
        self, markers: list[str] | None
    ) -> list[str] | None:
        """Expand current marker ids with the on-disk names a patch bump renamed.

        SQL tables still store pre-bump ids. Include those old names so a filter
        for the current id (``MarkerANew``) still matches stored ``MarkerA``.
        """
        if not markers:
            return markers
        remaps = self.marker_id_renames_by_sample()
        if not remaps or not any(remaps.values()):
            return markers
        stored = set(markers)
        requested = set(markers)
        for mapping in remaps.values():
            for old, new in mapping.items():
                if new in requested or old in requested:
                    stored.add(old)
        return list(stored)

    def current_marker_ids(self, markers: list[str]) -> list[str]:
        """Map requested marker ids through any applied panel patch rename."""
        remaps = self.marker_id_renames_by_sample()
        old_to_new: dict[str, str] = {}
        for mapping in remaps.values():
            old_to_new.update(mapping)
        return [old_to_new.get(marker, marker) for marker in markers]

    def _try_bump_adata_panel_version(
        self,
        adatas: list[AnnData],
        sample_names: list[str],
    ) -> list[AnnData]:
        """Try to bump the panel version of the given AnnData.

        Only try to upgrade to the latest version available in the view,
        if the panels differ in patch version and have the same product.

        ``sample_names`` must be aligned with ``adatas`` (one name per ``.pxl``).
        Each file is treated as a single sample or pool. Marker remaps are
        recorded under that file name so edgelist joins match the ``sample``
        column injected from the view.
        """
        if len(adatas) != len(sample_names):
            raise ValueError(
                "Panel bump requires one sample name per AnnData object, "
                f"got {len(adatas)} objects and {len(sample_names)} names."
            )
        self._marker_id_renames_by_sample = {}
        if any(
            (
                ("panel_metadata" not in adata.uns)
                and ("num_partial_panels" not in adata.uns)
            )
            for adata in adatas
        ):
            logger.debug(
                "Missing panel metadata in one or more samples, "
                + "skipping automatic panel version upgrade."
            )
            return adatas

        panel_combinations = [
            PNAAntibodyPanelCombination.from_adata(adata) for adata in adatas
        ]
        partial_panels: dict[str, dict[semver.Version, PartialPNAAntibodyPanel]] = (
            defaultdict(dict)
        )
        for pc in panel_combinations:
            for pp in pc.partial_panels():
                partial_panels[pp.metadata.product][
                    semver.Version.parse(pp.metadata.version)
                ] = pp
        for product, versions in partial_panels.items():
            if product is None:
                logger.debug(
                    "Found panels with missing product information, skipping automatic panel version upgrade for these panels."
                )
                continue
            latest_panel_version = max(versions.keys())
            if (
                all(ver.major == latest_panel_version.major for ver in versions.keys())
                and all(
                    ver.minor == latest_panel_version.minor for ver in versions.keys()
                )
                and not all(
                    ver.patch == latest_panel_version.patch for ver in versions.keys()
                )
            ):
                logger.info(
                    "Multiple panel patch versions for same product detected across samples. "
                    + "Attempting to upgrade to the latest."
                )

                latest_panel = versions[latest_panel_version]
                for i, (adata, pc, sample_name) in enumerate(
                    zip(adatas, panel_combinations, sample_names, strict=True)
                ):
                    panel = {pp.metadata.product: pp for pp in pc.partial_panels()}.get(
                        product
                    )
                    if panel is None:
                        continue
                    if panel.version != latest_panel.version:
                        diff = PNAAntibodyPanelDiff(panel, latest_panel)
                        adatas[i] = diff.upgrade_adata(adata)
                        mapping = diff.edgelist_marker_id_mapping(
                            collapsed=sample_calling_hashing_collapsed(adatas[i])
                        )
                        if mapping:
                            merged = dict(
                                self._marker_id_renames_by_sample.get(sample_name, {})
                            )
                            merged.update(mapping)
                            self._marker_id_renames_by_sample[sample_name] = merged
        return adatas

    def _read_adata_from_sample(
        self,
        *,
        session: PixelDataViewerSession,
        sample: str,
    ) -> AnnData:
        qb = QueryBuilder()
        db_name = self._view.normalized_sample_db_name(sample)

        # Read full AnnData contents (components/markers are filtered in-memory later).
        X = session.execute_eager(qb.adata_X_query(db_name, None)).to_pandas()
        var = session.execute_eager(qb.adata_var_query(db_name, None)).to_pandas()
        obs = session.execute_eager(qb.adata_obs_query(db_name, None)).to_pandas()

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
