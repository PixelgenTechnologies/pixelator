"""PNA antibody panel loading.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import pandas as pd
import polars as pl

from pixelator.common.config.panel import (
    AntibodyPanelMetadata,
    parse_panel_header_metadata,
)
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.hashing import _hashing_marker_ids
from pixelator.pna.config.panel.validation import (
    _INDEX_COLUMN,
    _INDEX_COLUMN_TYPE,
    _REQUIRED_COLUMNS,
    _UNIQUE_COLUMNS,
    validate_antibody_panel,
)

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


class PNAAntibodyPanel:
    """Class representing a PNA antibody panel."""

    _INDEX_COLUMN = _INDEX_COLUMN
    _INDEX_COLUMN_TYPE = _INDEX_COLUMN_TYPE
    _REQUIRED_COLUMNS = _REQUIRED_COLUMNS
    _UNIQUE_COLUMNS = _UNIQUE_COLUMNS

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> None:
        """Load a panel from a dataframe and metadata.

        Args:
            df: The dataframe containing the panel information.
            metadata: The metadata for the panel.
            file_name: The optional basename of the file from which the panel is loaded.
            filepath: The optional full path of the file from which the panel is loaded.

        Returns:
            None
        Raises:
            AssertionError: exception if panel file is missing, invalid or with incorrect format
        """
        self._filename = file_name
        self._filepath: Optional[Path] = Path(filepath).resolve() if filepath else None
        self.metadata = metadata
        self._df = df

        # validate the panel
        errors = self.validate_antibody_panel(df)
        if len(errors) > 0:
            msg_str = "\n".join(errors)
            raise AssertionError(
                f"The following errors were found validating the panel: {msg_str}"
            )

    @classmethod
    def from_csv(cls, filename: PathType) -> Self:
        """Create an AntibodyPanel from a csv panel file.

        Args:
            filename: The path to the panel file.

        Returns:
            The AntibodyPanel object. (AntibodyPanel)

        Raises:
            AssertionError: exception if panel file is missing,
        """
        panel_file = Path(filename)

        if not panel_file.is_file() or panel_file.suffix != ".csv":
            raise AssertionError(
                f"Panel file {filename} not found or has an incorrect format"
            )

        logger.debug("Creating Antibody panel from file %s", filename)

        df = cls._parse_panel(panel_file)
        metadata = cls._parse_header(panel_file)

        logger.debug("Antibody panel from file %s created", filename)

        return cls(df, metadata, file_name=panel_file.name, filepath=panel_file)

    @classmethod
    def from_pxl_dataset(
        cls, pxl_data: PNAPixelDataset, file_name: Optional[str] = None
    ) -> Self:
        """Create an AntibodyPanel from a pxl dataset.

        Args:
            pxl_data: A PNAPixelDataset object.
            file_name: The optional name of the file from which the pxl dataset was loaded.

        Returns:
            The AntibodyPanel object. (AntibodyPanel)

        Raises:
            KeyError: exception if panel information is missing in the pxl dataset,
        """
        logger.debug("Creating Antibody panel from PNAPixelDataset object")
        adata = pxl_data.adata()
        panel = cls.from_adata(adata, file_name=file_name)
        logger.debug("Antibody panel from PNAPixelDataset created")
        return panel

    @classmethod
    def from_adata(cls, adata: AnnData, file_name: Optional[str] = None) -> Self:
        """Create an AntibodyPanel from an AnnData object.

        Args:
            adata: An AnnData object containing panel information.
            file_name: The optional name of the file from which the AnnData object was loaded.

        Returns:
            The AntibodyPanel object. (AntibodyPanel)

        Raises:
            KeyError: exception if panel information is missing in the AnnData object.
        """
        logger.debug("Creating Antibody panel from AnnData object")
        try:
            panel_metadata = adata.uns["panel_metadata"]
        except KeyError as err:
            logger.error(  # pylint: disable=logging-not-lazy
                f"The provided AnnData object does not contain {err}. "
                + "Please, regenerate your data with the most recent version of pixelator."
            )
            raise
        panel_columns = panel_metadata.get("panel_columns")
        if not panel_columns:
            raise KeyError(
                "The provided AnnData object does not contain panel columns information in the metadata. "
                + "Please, regenerate your data with the most recent version of pixelator."
            )
        df = adata.var[panel_columns]
        metadata = AntibodyPanelMetadata.model_validate(panel_metadata)

        logger.debug("Antibody panel from AnnData object created")
        return cls(df, metadata, file_name=file_name)

    @property
    def name(self) -> str:
        """Panel name from metadata.

        Returns:
            The panel name.
        """
        return self.metadata.name

    @property
    def product(self) -> Optional[str]:
        """Product identifier from metadata, if present.

        Returns:
            Product name, or None when not provided in panel metadata.
        """
        return self.metadata.product

    @property
    def version(self) -> str:
        """Panel version from metadata.

        Returns:
            Semantic version string for this panel.
        """
        return self.metadata.version

    @property
    def description(self) -> Optional[str]:
        """Return the panel file description."""
        return self.metadata.description

    @property
    def aliases(self) -> list[str]:
        """Return the (optional) list of panel file aliases."""
        return self.metadata.aliases

    @property
    def archived(self) -> Optional[bool]:
        """Return whether the panel is marked as archived."""
        return self.metadata.archived

    @property
    def hashing_marker_ids(self) -> set[str]:
        """Return marker ids flagged by the ``sample_hashing`` column."""
        return _hashing_marker_ids(self.df)

    @classmethod
    def _parse_header(cls, file: Path) -> AntibodyPanelMetadata:
        """Parse front-matter YAML metadata from a panel file.

        Args:
            file: Panel CSV file whose leading comment block contains YAML metadata.

        Returns:
            Parsed panel metadata.

        Raises:
            ValueError: If no metadata header is present in the file.
        """
        return parse_panel_header_metadata(file)

    @classmethod
    def _parse_panel(cls, panel_file: Path) -> pd.DataFrame:
        panel = pd.read_csv(str(panel_file), comment="#", index_col="marker_id").fillna(
            ""
        )

        panel["control"] = panel["control"].map(lambda s: s.lower() == "yes")

        return panel.copy()

    @cached_property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self._df[self._df["control"]].index)

    @cached_property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self._df.index.unique())

    @property
    def df(self) -> pd.DataFrame:
        """Return the panel dataframe."""
        return self._df

    @property
    def filename(self) -> Optional[str]:
        """Return the filename of the marker panel."""
        return self._filename

    @property
    def filepath(self) -> Optional[Path]:
        """Return the full path of the marker panel file, if any."""
        return self._filepath

    @cached_property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self._df.shape[0]

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
        return validate_antibody_panel(panel_df, validate_types=validate_types)

    def to_polars(self) -> pl.DataFrame:
        """Convert the panel to a Polars DataFrame."""
        return pl.from_pandas(self.df, include_index=True)

    def __eq__(self, other: object) -> bool:
        """Check if two panels are equal based on their dataframes and metadata.

        Args:
            other: Panel to compare for equality.
        """
        if not isinstance(other, PNAAntibodyPanel):
            raise ValueError("Can only compare with another PNAAntibodyPanel")
        return self.df.equals(other.df) and self.metadata == other.metadata


def load_antibody_panel(config: PNAConfig, panel: PathType) -> PNAAntibodyPanel:
    """Load an antibody panel from a file or from the config file.

    Args:
        config: the config object
        panel: the path to the panel file or the name of the panel in the config file
    Returns:
        the antibody panel (PNAAntibodyPanel)
    """
    panel_str = str(panel)
    panel_from_config = config.get_panel(panel_str)

    if panel_from_config is not None:
        logger.info("Found panel in config file: %s", panel_from_config.name)
        return panel_from_config

    panel_obj = PNAAntibodyPanel.from_csv(panel)
    return panel_obj
