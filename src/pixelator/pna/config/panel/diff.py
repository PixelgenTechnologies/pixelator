"""Comparison and patch-level upgrades between two PNA antibody panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from functools import cached_property
from typing import List, Set

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.common.config.panel import AntibodyPanelMetadata
from pixelator.common.utils import logger
from pixelator.pna.config.panel.combination import PNAAntibodyPanelCombination
from pixelator.pna.config.panel.partial import PartialPNAAntibodyPanel
from pixelator.pna.config.panel.utils import (
    collapsed_hashing_marker_id,
    split_hashing_marker_id,
)
from pixelator.pna.utils.sample_calling_uns import (
    ORIGINAL_HASH_COUNTS_PREFIX,
    sample_calling_hashing_collapsed,
)


class PNAAntibodyPanelDiff:
    """Compare two :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel` instances by clone sequences.

    Used to describe column-level differences between panel versions and to
    support automatic AnnData panel patch upgrades via :meth:`upgrade_adata`.
    """

    join_on_columns: list[str] = ["sequence_1", "sequence_2"]

    def __init__(
        self, panel_1: PartialPNAAntibodyPanel, panel_2: PartialPNAAntibodyPanel
    ) -> None:
        """Initialize the PNAAntibodyPanelDiff object.

        Args:
            panel_1: The first panel to compare.
            panel_2: The second panel to compare.
        """
        self.panel_1 = panel_1
        self.panel_2 = panel_2

        logger.debug(
            "Comparing panels %s v%s and %s v%s",
            panel_1.name,
            panel_1.version,
            panel_2.name,
            panel_2.version,
        )

        self.joined = self.panel_1.to_polars().join(
            self.panel_2.to_polars(),
            on=self.join_on_columns,
            how="full",
            suffix="_panel_2",
            maintain_order="left",
        )

        self._identical_columns: List[str] | None = None
        self._changed_columns: List[str] | None = None
        self._removed_columns: Set[str] | None = None
        self._added_columns: Set[str] | None = None

    @property
    def col_names_in_both_panels(self) -> List[str]:
        """Return a list of column names that are present in both panels."""
        return list(
            set(self.panel_1.to_polars().columns).intersection(
                set(self.panel_2.to_polars().columns)
            )
        )

    @property
    def identical_columns(self) -> List[str]:
        """Return a list of columns that are identical between the two panels."""
        return [
            col_name
            for col_name in self.col_names_in_both_panels
            if self.joined[col_name]
            .eq_missing(self.joined[col_name + "_panel_2"])
            .all()
        ]

    @cached_property
    def changed_columns(self) -> List[str]:
        """Return a list of columns that are different between the two panels."""
        changed_columns = [
            col_name
            for col_name in set(self.col_names_in_both_panels).difference(
                set(self.join_on_columns)
            )
            if not self.joined[col_name]
            .eq_missing(self.joined[col_name + "_panel_2"])
            .all()
        ]
        for col_name in changed_columns:
            diff_count = self.joined.filter(
                pl.col(col_name).ne_missing(pl.col(col_name + "_panel_2"))
            ).shape[0]
            logger.debug(
                "Column %s is different between the two panels %s and %s (%d differing entries).",
                col_name,
                self.panel_1.name,
                self.panel_2.name,
                diff_count,
            )
        return changed_columns

    @cached_property
    def removed_columns(self) -> List[str]:
        """Return a list of columns that are present in panel 1 but not in panel 2."""
        removed_columns = set(self.panel_1.to_polars().columns).difference(
            set(self.panel_2.to_polars().columns)
        )
        for col_name in removed_columns:
            logger.debug(
                "Column %s is present in panel %s but not in panel %s.",
                col_name,
                self.panel_1.name,
                self.panel_2.name,
            )
        return sorted(removed_columns)

    @cached_property
    def added_columns(self) -> List[str]:
        """Return a list of columns that are present in panel 2 but not in panel 1."""
        added_columns = set(self.panel_2.to_polars().columns).difference(
            set(self.panel_1.to_polars().columns)
        )
        for col_name in added_columns:
            logger.debug(
                "Column %s is present in panel %s but not in panel %s.",
                col_name,
                self.panel_2.name,
                self.panel_1.name,
            )
        return sorted(added_columns)

    @property
    def added_clones(self) -> pl.DataFrame:
        """Return a dataframe with the clones that are present in panel 2 but not in panel 1."""
        return (
            self.joined.filter(
                pl.any_horizontal(
                    pl.col(col_name).is_null()
                    & pl.col(col_name + "_panel_2").is_not_null()
                    for col_name in self.join_on_columns
                )
            )
            .drop([col_name for col_name in self.panel_1.to_polars().columns])
            .rename(
                {
                    col_name + "_panel_2": col_name
                    for col_name in self.panel_2.to_polars().columns
                    if col_name + "_panel_2" in self.joined.columns
                }
            )
        )

    @property
    def removed_clones(self) -> pl.DataFrame:
        """Return a dataframe with the clones that are present in panel 1 but not in panel 2."""
        return self.joined.filter(
            pl.any_horizontal(
                pl.col(col_name).is_not_null() & pl.col(col_name + "_panel_2").is_null()
                for col_name in self.join_on_columns
            )
        ).drop(
            [
                col_name + "_panel_2"
                if col_name in self.joined.columns
                and col_name not in self.added_columns
                else col_name
                for col_name in self.panel_2.to_polars().columns
            ]
        )

    def marker_id_mapping(self) -> dict[str, str]:
        """Return ``panel_1`` marker_id → ``panel_2`` marker_id for renamed clones.

        Clones that keep the same name, or that are only present in one panel,
        are omitted.
        """
        if "marker_id" not in self.joined.columns:
            return {}
        new_col = (
            "marker_id_panel_2" if "marker_id_panel_2" in self.joined.columns else None
        )
        if new_col is None:
            return {}
        mapping: dict[str, str] = {}
        for old, new in self.joined.select("marker_id", new_col).iter_rows():
            if old is None or new is None or old == new:
                continue
            mapping[str(old)] = str(new)
        return mapping

    def _iter_hashing_marker_id_pairs(self) -> list[tuple[str, str]]:
        """Return ``(panel_1, panel_2)`` marker ids for ``sample_hashing`` clones."""
        hashing_ids = self.panel_1.hashing_marker_ids
        if not hashing_ids or "marker_id" not in self.joined.columns:
            return []
        new_col = (
            "marker_id_panel_2" if "marker_id_panel_2" in self.joined.columns else None
        )
        if new_col is None:
            return []
        pairs: list[tuple[str, str]] = []
        for old, new in self.joined.select("marker_id", new_col).iter_rows():
            if old is None or new is None:
                continue
            if str(old) not in hashing_ids:
                continue
            pairs.append((str(old), str(new)))
        return pairs

    def _validate_hashing_marker_id_renames(self) -> None:
        """Reject hashing marker_id changes that alter the hash-group suffix.

        ``B2M-1`` may become ``NEWB2MNAME-1``. Changing ``-1`` is not allowed.
        All hashing ids that share a collapsed base name must map to the same
        new base.
        """
        base_to_new_bases: dict[str, set[str]] = {}
        for old, new in self._iter_hashing_marker_id_pairs():
            if old != new:
                old_parts = split_hashing_marker_id(old)
                new_parts = split_hashing_marker_id(new)
                if old_parts is None:
                    raise ValueError(
                        "Hashing marker ids must end with -<digits> to be renamed "
                        f"in a panel patch bump, got {old!r} -> {new!r}."
                    )
                if new_parts is None or new_parts[1] != old_parts[1]:
                    old_suffix = f"-{old_parts[1]}"
                    raise ValueError(
                        "Hashing marker rename may only change the base name, not "
                        f"the hash group suffix {old_suffix}. Got {old!r} -> {new!r}."
                    )
            old_base = collapsed_hashing_marker_id(old)
            new_base = collapsed_hashing_marker_id(new)
            base_to_new_bases.setdefault(old_base, set()).add(new_base)
        for old_base, new_bases in base_to_new_bases.items():
            if len(new_bases) != 1:
                raise ValueError(
                    "Hashing markers with collapsed name "
                    f"{old_base!r} map to multiple names {sorted(new_bases)}. "
                    "A patch bump must keep a single base name per hash group family."
                )

    def collapsed_hashing_marker_id_mapping(self) -> dict[str, str]:
        """Return collapsed hashing names ``panel_1`` → ``panel_2``.

        ``B2M-1`` / ``B2M-2`` collapsing to ``B2M`` become ``NEWB2MNAME`` when
        those clones are renamed to ``NEWB2MNAME-1`` / ``NEWB2MNAME-2``. All
        hashing ids that share an old base must map to the same new base.
        """
        self._validate_hashing_marker_id_renames()
        mapping: dict[str, str] = {}
        for old, new in self._iter_hashing_marker_id_pairs():
            old_base = collapsed_hashing_marker_id(old)
            new_base = collapsed_hashing_marker_id(new)
            if old_base != new_base:
                mapping[old_base] = new_base
        return mapping

    def edgelist_marker_id_mapping(self, *, collapsed: bool = False) -> dict[str, str]:
        """Marker remaps to apply to edgelist ``marker_1`` / ``marker_2``.

        When ``collapsed`` is true, hashing ids are already stored under their
        base name, so the collapsed-name mapping is merged in.

        The pixel-file read path uses this mapping via
        :class:`~pixelator.pna.pixeldataset.io.anndata_helper.AnnDataHelper`.
        The edgelist reader then applies it, joining on ``sample`` when that
        column is present.
        """
        mapping = dict(self.marker_id_mapping())
        if collapsed:
            mapping.update(self.collapsed_hashing_marker_id_mapping())
        return mapping

    def _rename_original_hash_counts(self, adata: AnnData) -> None:
        """Rename ``original_hash_counts_{id}`` columns when hashing ids change."""
        hashing_ids = self.panel_1.hashing_marker_ids
        rename: dict[str, str] = {}
        for old, new in self.marker_id_mapping().items():
            if old not in hashing_ids or old == new:
                continue
            old_col = f"{ORIGINAL_HASH_COUNTS_PREFIX}{old}"
            new_col = f"{ORIGINAL_HASH_COUNTS_PREFIX}{new}"
            if old_col not in adata.obs.columns:
                continue
            if new_col in adata.obs.columns:
                raise ValueError(
                    "Cannot rename hashing counts column "
                    f"{old_col!r} to {new_col!r}: the target already exists."
                )
            rename[old_col] = new_col
        if rename:
            adata.obs.rename(columns=rename, inplace=True)

    def _apply_collapsed_var_index_renames(self, adata: AnnData) -> None:
        """Rename collapsed hashing marker ids in ``adata.var``."""
        collapsed_map = self.collapsed_hashing_marker_id_mapping()
        if not collapsed_map:
            return
        new_index = pd.Index(
            [collapsed_map.get(str(name), str(name)) for name in adata.var.index],
            name=adata.var.index.name,
        )
        if new_index.has_duplicates:
            duplicated = new_index[new_index.duplicated()].unique().tolist()
            raise ValueError(
                "Collapsed hashing marker rename would duplicate var index "
                f"names {duplicated}."
            )
        adata.var.index = new_index

    def upgrade_adata(self, adata: AnnData) -> AnnData:
        """Apply a patch-level panel upgrade to AnnData marker annotations.

        Updates overlapping clone rows from ``panel_2`` when the two panels
        share the same product and only patch-compatible differences exist.

        Original panel snapshots in ``uns`` are always updated. ``var`` is
        patched only for clones that are present there. After sample calling,
        hashing clones are expected to be absent from ``var``; missing hashing
        clones then do not raise. That layout is taken from
        ``uns["sample_calling"]["collapsed"]`` when the key is present, and
        otherwise inferred from ``original_hash_counts_*`` on ``obs`` or from
        hashing clones missing in ``var``. Missing non-hashing clones
        still raise.
        Hashing ``marker_id`` bumps may only change the base name
        (``B2M-1`` → ``NEWB2MNAME-1``), never the hash group. That rename is
        applied to ``original_hash_counts_*`` and, when collapsed, to the
        collapsed marker id in ``var``.

        Args:
            adata: AnnData whose ``var`` table embeds panel columns.

        Returns:
            The same AnnData instance after in-place panel column updates.

        Raises:
            ValueError: If clones were added/removed, products differ, panels
                do not match the AnnData contents, row counts diverge, a
                non-hashing clone is missing from ``var``, or a hashing
                marker_id rename changes the hash-group suffix.
        """
        if len(self.added_clones) > 0:
            raise ValueError(
                "Cannot automatically upgrade panel if there are added clones. "
                + "Please check the differences between the panels and upgrade manually."
            )
        if len(self.removed_clones) > 0:
            raise ValueError(
                "Cannot automatically upgrade panel if there are removed clones. "
                + "Please check the differences between the panels and upgrade manually."
            )

        adata_panel = (
            [
                pp
                for pp in PNAAntibodyPanelCombination.from_adata(adata).partial_panels()
                if pp.metadata.name == self.panel_1.name
                and pp.metadata.version == self.panel_1.version
            ]
            or [None]
        ).pop()
        if adata_panel is None:
            raise ValueError(
                "The provided AnnData object does not contain the panel. Cannot upgrade."
            )
        if self.panel_1 != adata_panel:
            raise ValueError(
                "The provided AnnData object does not match the panel. Cannot upgrade."
                + f" Expected panel {self.panel_2.name} v{self.panel_2.version}, "
                + f"but got panel {adata_panel.name} v{adata_panel.version}."
            )

        self._validate_hashing_marker_id_renames()

        # first update the uns variables
        if "num_partial_panels" in adata.uns:
            for idx in range(adata.uns["num_partial_panels"]):
                metadata_key = f"panel_metadata__{idx}"
                metadata = AntibodyPanelMetadata.model_validate(adata.uns[metadata_key])
                if (
                    metadata.name == self.panel_1.name
                    and metadata.version == self.panel_1.version
                ):
                    adata.uns[metadata_key] = self.panel_2.metadata.to_dict()
                    adata.uns[f"panel_df__{idx}"] = self.panel_2.df.to_csv()
                    break
        elif "panel_metadata" in adata.uns:
            # Migrate legacy single-panel uns keys to the multi-panel layout.
            del adata.uns["panel_metadata"]
            adata.uns.pop("panel_columns", None)
            adata.uns["num_partial_panels"] = 1
            adata.uns["panel_metadata__0"] = self.panel_2.metadata.to_dict()
            adata.uns["panel_df__0"] = self.panel_2.df.to_csv()

        # update the anndata var table
        org_var_shape = adata.var.shape
        org_index = adata.var.index.name
        collapsed = sample_calling_hashing_collapsed(adata)
        adata.var.reset_index(inplace=True)
        panel1_pl = self.panel_1.to_polars()
        panel1_row_identifiers = list(
            panel1_pl.select(self.join_on_columns).iter_rows()
        )
        panel1_marker_ids = [str(marker_id) for marker_id in panel1_pl["marker_id"]]
        var_identifiers = list(
            adata.var[self.join_on_columns].itertuples(index=False, name=None)
        )
        var_id_to_pos = {ident: i for i, ident in enumerate(var_identifiers)}
        hashing_marker_ids = self.panel_1.hashing_marker_ids
        panel_rows_present: list[int] = []
        row_indexes_in_var: list[int] = []
        missing_non_hashing: list[tuple] = []
        for panel_row, ident in enumerate(panel1_row_identifiers):
            var_pos = var_id_to_pos.get(ident)
            if var_pos is None:
                marker_id = panel1_marker_ids[panel_row]
                if collapsed and marker_id in hashing_marker_ids:
                    continue
                missing_non_hashing.append(ident)
                continue
            panel_rows_present.append(panel_row)
            row_indexes_in_var.append(var_pos)

        if missing_non_hashing:
            raise ValueError(
                "The provided AnnData object is missing panel clones required "
                "for a 1:1 var upgrade. After sample calling, hashing clones "
                "are expected to be absent from var when the layout is "
                "collapsed (uns['sample_calling']['collapsed'], or inferred "
                "from missing hashing clones / original_hash_counts_*) and "
                "that is not an error; missing non-hashing clones still raise. "
                f"Missing sequence pairs: {missing_non_hashing[:5]}"
            )

        present_idents = {panel1_row_identifiers[i] for i in panel_rows_present}
        is_panel_row = np.array([ident in present_idents for ident in var_identifiers])
        var_row_indices = np.array(row_indexes_in_var, dtype=int)
        panel_row_indices = np.array(panel_rows_present, dtype=int)

        # define local helper to avoid repeated code,
        # uses adata and var_row_indices from outer scope
        def _copy_joined_to_var(column_name: str, joined_column: str) -> None:
            if var_row_indices.size == 0:
                return
            values = self.joined[joined_column].to_numpy()[panel_row_indices]
            adata.var.iloc[var_row_indices, adata.var.columns.get_loc(column_name)] = (
                values
            )

        for col in self.removed_columns:
            if (adata.var.loc[~is_panel_row, col].fillna("") == "").all():
                adata.var.drop(col, inplace=True, errors="ignore", axis=1)
            else:
                adata.var.loc[is_panel_row, col] = pd.NA
        for col in self.added_columns:
            if col not in adata.var.columns:
                adata.var[col] = pd.NA
            else:
                assert (adata.var.loc[is_panel_row, col].fillna("") == "").all(), (
                    "added column already exists in adata.var with non-empty values for some of the"
                    + " panel rows. Cannot automatically upgrade."
                )
            _copy_joined_to_var(col, col)
        for col in self.changed_columns:
            _copy_joined_to_var(col, col + "_panel_2")
        adata.var.set_index(org_index, inplace=True)

        self._rename_original_hash_counts(adata)
        if collapsed:
            self._apply_collapsed_var_index_renames(adata)

        # shape check
        if adata.var.shape[0] != org_var_shape[0]:
            raise ValueError(
                "Row count mismatch in automatic patch panel patch version bump."
            )
        return adata
