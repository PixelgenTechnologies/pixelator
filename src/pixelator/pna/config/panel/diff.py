"""Differences between two PNA antibody panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from functools import cached_property
from typing import List, Set

import polars as pl
from anndata import AnnData

from pixelator.common.utils import logger
from pixelator.pna.config.panel.antibody_panel import PNAAntibodyPanel


class PNAAntibodyPanelDiff:
    """Class representing the differences between two PNAAntibodyPanel objects."""

    join_on_columns: list[str] = ["sequence_1", "sequence_2"]

    def __init__(self, panel_1: PNAAntibodyPanel, panel_2: PNAAntibodyPanel) -> None:
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
        """Upgrade an AnnData object with the changes between the two panels.

        Args:
            adata: An AnnData object containing panel information.
        """
        adata_panel = PNAAntibodyPanel.from_adata(adata)
        if self.panel_1 != adata_panel:
            raise ValueError(
                "The provided AnnData object does not match the panel. Cannot upgrade."
                f"Expected panel {self.panel_2.name} v{self.panel_2.version}, but got panel {adata_panel.name} v{adata_panel.version}."
            )

        non_panel_columns = adata.var.copy()[
            [
                col
                for col in adata.var.columns
                if col not in adata.uns["panel_metadata"]["panel_columns"]
            ]
            + self.join_on_columns
        ]
        adata.var = (
            self.joined.select(
                list(
                    set(
                        self.join_on_columns
                        + self.identical_columns
                        + [f"{col}_panel_2" for col in self.changed_columns]
                        + self.added_columns
                    )
                )
            )
            .rename({f"{col}_panel_2": col for col in self.changed_columns})
            # keep order and append new to the end
            .select(
                ["marker_id"]  # index not in panel_metadata panel_columns below
                + adata.uns["panel_metadata"]["panel_columns"]
                + self.added_columns
            )
            .to_pandas()
            .set_index("marker_id")
        )
        if adata.var.shape[0] != non_panel_columns.shape[0]:
            raise ValueError(
                "Row count mismatch in automatic patch panel patch version bump."
            )
        adata.var = adata.var.join(
            non_panel_columns.set_index(self.join_on_columns),
            how="outer",
            on=self.join_on_columns,
        )
        adata.uns["panel_metadata"] = self.panel_2.metadata.model_dump()
        adata.uns["panel_metadata"]["panel_columns"] = self.panel_2.df.columns.tolist()
        return adata
