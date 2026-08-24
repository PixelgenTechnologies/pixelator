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

    def upgrade_adata(self, adata: AnnData) -> AnnData:
        """Apply a patch-level panel upgrade to AnnData marker annotations.

        Updates overlapping clone rows from ``panel_2`` when the two panels
        share the same product and only patch-compatible differences exist.

        Args:
            adata: AnnData whose ``var`` table embeds panel columns.

        Returns:
            The same AnnData instance after in-place panel column updates.

        Raises:
            ValueError: If clones were added/removed, products differ, panels
                do not match the AnnData contents, or row counts diverge.
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

        # update the andata var table
        org_var_shape = adata.var.shape
        org_index = adata.var.index.name
        adata.var.reset_index(inplace=True)
        panel1_row_identifiers = list(
            self.panel_1.to_polars().select(self.join_on_columns).iter_rows()
        )
        var_identifiers = list(
            adata.var[self.join_on_columns].itertuples(index=False, name=None)
        )
        is_panel_row = np.array(
            [row in panel1_row_identifiers for row in var_identifiers]
        )
        row_indexes_in_var = np.array(
            [var_identifiers.index(row) for row in panel1_row_identifiers]
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
            adata.var.iloc[row_indexes_in_var, adata.var.columns.get_loc(col)] = (
                self.joined[col]
            )
        for col in self.changed_columns:
            adata.var.iloc[row_indexes_in_var, adata.var.columns.get_loc(col)] = (
                self.joined[col + "_panel_2"]
            )
        adata.var.set_index(org_index, inplace=True)

        # shape check
        if adata.var.shape[0] != org_var_shape[0]:
            raise ValueError(
                "Row count mismatch in automatic patch panel patch version bump."
            )
        return adata
