"""Shared read interface for PNA panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import polars as pl
from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pixelator.common.config.panel import AntibodyPanelMetadata
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.utils import (
    _resolve_panel_source_from_pxl,
    nested_hashing_marker_ids,
    sample_hashing_mask,
    split_hashing_marker_id,
)

if TYPE_CHECKING:
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


class PNAPanel(ABC):
    """Shared read interface for single PNA panels and panel combinations.

    Subclasses must provide :attr:`df` (and typically :meth:`from_adata`).
    Marker helpers, size, Polars conversion, and schema validation are
    implemented here in terms of that dataframe.
    """

    _INDEX_COLUMN = "marker_id"
    _INDEX_COLUMN_TYPE = str
    _REQUIRED_COLUMNS = {
        "control": bool,
        "sequence_1": str,
        "sequence_2": str,
    }
    _UNIQUE_COLUMNS = ["sequence_1", "sequence_2"]

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Return the panel marker dataframe indexed by ``marker_id``."""

    @property
    @abstractmethod
    def metadata(self) -> AntibodyPanelMetadata | list[AntibodyPanelMetadata]:
        """Return panel metadata (one entry, or one per member for combinations)."""

    @classmethod
    @abstractmethod
    def from_adata(
        cls,
        adata: AnnData,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a panel instance from AnnData-embedded panel data.

        Args:
            adata: AnnData with embedded panel information.
            file_name: Optional basename of the source file.
            filepath: Optional full path of the source file.
        """

    @classmethod
    def from_pxl_dataset(
        cls,
        pxl_data: PNAPixelDataset,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a panel from a PNA pixel dataset via :meth:`from_adata`.

        When ``file_name`` / ``filepath`` are omitted and ``pxl_data`` wraps a
        single ``.pxl`` file, those values are taken from that file path.

        Args:
            pxl_data: Dataset whose ``adata()`` holds panel information.
            file_name: Optional source file basename to attach to the panel.
            filepath: Optional full source path to attach to the panel.

        Returns:
            A panel of this class.
        """
        logger.debug("Creating panel from PNAPixelDataset object")
        file_name, filepath = _resolve_panel_source_from_pxl(
            pxl_data, file_name=file_name, filepath=filepath
        )
        panel = cls.from_adata(pxl_data.adata(), file_name=file_name, filepath=filepath)
        logger.debug("Panel from PNAPixelDataset created")
        return panel

    @property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self.df[self.df["control"]].index)

    @property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self.df.index.unique())

    @property
    def hashing_marker_ids(self) -> set[str]:
        """Return ``marker_id`` values flagged by the ``sample_hashing`` column.

        Names such as ``PD-1`` or ``TIM-3`` are not hashing markers unless
        that column is true for the row. Combinations use the concatenated
        ``df``, so hashing rows on a base panel and members of a hashing
        panel are both included.
        """
        df = self.df
        if "sample_hashing" not in df.columns:
            return set()
        mask = sample_hashing_mask(df["sample_hashing"])
        return {str(marker_id) for marker_id in df.index[mask]}

    @property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self.df.shape[0]

    def to_polars(self) -> pl.DataFrame:
        """Convert the panel to a Polars DataFrame."""
        return pl.from_pandas(self.df, include_index=True)

    @staticmethod
    def _validate_sequences(panel_df, sequence_col):
        errors = []
        sequences = panel_df[sequence_col]
        ref_length = len(sequences.iloc[0])
        if not sequences.apply(lambda x: len(x) == ref_length).all():
            errors.append(f"All {sequence_col} values must have the same length.")

        if not sequences.str.match("^[ATCG]*$").all():
            errors.append(
                f"All {sequence_col} values must only contain ATCG characters. Offending values: "
                f"{sequences[~sequences.str.match('^[ATCG]*$')].tolist()}"
            )

        return errors

    @staticmethod
    def _validate_marker_names(panel_df):
        errors = []
        if any(panel_df.index.str.contains("_")):
            # Markers should not contain underscores since this messes
            # things up with Seurat on the R side
            errors.append(
                "The marker_id column should not contain underscores. "
                "Please use dashes instead. Offending values: "
                f"{panel_df.index[panel_df.index.str.contains('_')]}"
            )
        if any(panel_df.index.str.contains(r"\s")):
            # Markers should not contain white-spaces since this causes
            # issues in the demultiplexing step (and other places that
            # might assume that marker names are single tokens)
            problematic_lines = panel_df.index[panel_df.index.str.contains(r"\s")]
            errors.append(
                "The marker_id column should not contain white-spaces. "
                "Please use dashes instead or remove the white-spaces. Offending values: "
                f"{problematic_lines}"
            )
        return errors

    @staticmethod
    def _validate_hashing_marker_id_suffixes(panel_df: pd.DataFrame) -> list[str]:
        if "sample_hashing" not in panel_df.columns:
            return []
        mask = sample_hashing_mask(panel_df["sample_hashing"])
        missing_suffix = [
            str(marker_id)
            for marker_id in panel_df.index[mask]
            if split_hashing_marker_id(str(marker_id)) is None
        ]
        errors: list[str] = []
        if missing_suffix:
            errors.append(
                "Hashing marker ids must end with -<digits> (e.g. B2M-1). "
                f"Offending values: {missing_suffix}"
            )
        nested = nested_hashing_marker_ids(
            str(marker_id) for marker_id in panel_df.index[mask]
        )
        if nested:
            errors.append(
                "Hashing marker ids must not collapse to another hashing id "
                f"(e.g. B2M-1-1 next to B2M-1). Offending values: {nested}"
            )
        return errors

    @classmethod
    def validate_antibody_panel(
        cls, panel_df: pd.DataFrame, validate_types: bool = True
    ) -> list[str]:
        """Validate antibody panel schema and content.

        Args:
            panel_df: Dataframe containing panel markers and sequences.
            validate_types: If True, validate dataframe column types.

        Returns:
            A list of validation error messages. Empty means valid input.
        """
        errors = []

        # some basic sanity check on the panel size and columns
        if not set(cls._REQUIRED_COLUMNS).issubset(set(panel_df.columns)):
            missing_columns = set(cls._REQUIRED_COLUMNS) - set(panel_df.columns)
            errors.append(f"Panel has missing required columns: {missing_columns}")
            return errors

        if validate_types:
            panel_pl_df = pl.from_pandas(panel_df, include_index=True)
            for col, expected_type in (
                cls._REQUIRED_COLUMNS | {cls._INDEX_COLUMN: cls._INDEX_COLUMN_TYPE}
            ).items():
                found_type = panel_pl_df[col].dtype.to_python()
                if not found_type == expected_type:
                    errors.append(
                        f"Column {col} has incorrect type. "
                        + f"Expected {expected_type}, got {found_type}"
                    )

        if panel_df.shape[0] == 0:
            errors.append("Panel file is empty")
            return errors

        # sanity check on the unique columns
        for col in cls._UNIQUE_COLUMNS:
            if not len(panel_df[col].unique()) == len(panel_df[col]):
                errors.append(f"All values in column: {col} were not unique")

        if panel_df.index.name != cls._INDEX_COLUMN:
            errors.append(f"`{cls._INDEX_COLUMN}` is missing or is not set as index")
            return errors

        errors += cls._validate_marker_names(panel_df)
        errors += cls._validate_hashing_marker_id_suffixes(panel_df)

        if panel_df["control"].dtype != bool:
            errors.append("`control` column is not boolean")

        # Check UniProt IDs format conforming to the UniProt naming convention.
        # Empty IDs are allowed.
        if "uniprot_id" in panel_df.columns:
            # Pattern for valid UniProt IDs
            pattern = r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|$"

            def check_id(id_str):
                """Check id.

                Args:
                    id_str: id str.
                """
                return all(
                    bool(re.match(pattern, id_)) for id_ in str(id_str).split(";")
                )

            bad_ids = panel_df[~panel_df["uniprot_id"].apply(check_id)]["uniprot_id"]

            if len(bad_ids) > 0:
                errors.append(
                    "Invalid UniProt IDs found."
                    "Please conform to the naming convention or remove the following IDs:"
                    f"{bad_ids.tolist()}"
                )

        errors += cls._validate_sequences(panel_df, "sequence_1")
        errors += cls._validate_sequences(panel_df, "sequence_2")

        return errors
