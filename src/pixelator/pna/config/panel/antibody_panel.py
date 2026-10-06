"""PNA antibody panel loading and validation.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
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
from pixelator.pna.config.panel.hashing import (
    _hashing_marker_ids,
    collapsed_hashing_marker_id,
    split_hashing_marker_id,
)

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset

# UniProt accession naming convention. The trailing alternative allows an empty id.
_UNIPROT_ID_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|$"
)


def _uniprot_ids_are_valid(id_str: object) -> bool:
    """Return whether every semicolon-separated UniProt id matches the convention."""
    return all(bool(_UNIPROT_ID_RE.match(part)) for part in str(id_str).split(";"))


class PNAAntibodyPanel:
    """Class representing a PNA antibody panel."""

    # required columns
    _INDEX_COLUMN = "marker_id"
    _INDEX_COLUMN_TYPE = str
    _REQUIRED_COLUMNS = {
        "control": bool,
        "sequence_1": str,
        "sequence_2": str,
    }

    # and these should have unique values
    _UNIQUE_COLUMNS = ["sequence_1", "sequence_2"]

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

    @classmethod
    def _validate_structure(cls, panel_df: pd.DataFrame) -> list[str]:
        """Return the first error that makes the remaining checks meaningless."""
        if not set(cls._REQUIRED_COLUMNS).issubset(set(panel_df.columns)):
            missing_columns = set(cls._REQUIRED_COLUMNS) - set(panel_df.columns)
            return [f"Panel has missing required columns: {missing_columns}"]

        if panel_df.shape[0] == 0:
            return ["Panel file is empty"]

        if panel_df.index.name != cls._INDEX_COLUMN:
            return [f"`{cls._INDEX_COLUMN}` is missing or is not set as index"]

        return []

    @classmethod
    def _validate_column_types(cls, panel_df: pd.DataFrame) -> list[str]:
        """Return errors for required columns whose dtype does not match the schema."""
        errors = []
        panel_pl_df = pl.from_pandas(panel_df, include_index=True)
        for col, expected_type in (
            cls._REQUIRED_COLUMNS | {cls._INDEX_COLUMN: cls._INDEX_COLUMN_TYPE}
        ).items():
            found_type = panel_pl_df[col].dtype.to_python()
            if not found_type == expected_type:
                errors.append(
                    f"Column {col} has incorrect type. Expected {expected_type}, got {found_type}"
                )
        return errors

    @classmethod
    def _validate_unique_values(cls, panel_df: pd.DataFrame) -> list[str]:
        """Return errors when a unique column or the marker id repeats."""
        errors = []
        for col in cls._UNIQUE_COLUMNS:
            if not len(panel_df[col].unique()) == len(panel_df[col]):
                errors.append(f"All values in column: {col} were not unique")

        if panel_df.index.duplicated().any():
            duplicated = panel_df.index[panel_df.index.duplicated()].unique().tolist()
            errors.append(
                "All values in column: marker_id were not unique. "
                f"Offending values: {duplicated}"
            )
        return errors

    @staticmethod
    def _validate_control_column(panel_df: pd.DataFrame) -> list[str]:
        """Return an error when the control column is not boolean."""
        if panel_df["control"].dtype != bool:
            return ["`control` column is not boolean"]
        return []

    @staticmethod
    def _validate_uniprot_ids(panel_df: pd.DataFrame) -> list[str]:
        """Return an error when a present UniProt id breaks the naming convention."""
        if "uniprot_id" not in panel_df.columns:
            return []

        bad_ids = panel_df[~panel_df["uniprot_id"].apply(_uniprot_ids_are_valid)][
            "uniprot_id"
        ]
        if len(bad_ids) > 0:
            return [
                "Invalid UniProt IDs found."
                "Please conform to the naming convention or remove the following IDs:"
                f"{bad_ids.tolist()}"
            ]
        return []

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
        errors = cls._validate_structure(panel_df)
        if errors:
            return errors
        if validate_types:
            errors += cls._validate_column_types(panel_df)
        errors += cls._validate_unique_values(panel_df)
        errors += cls._validate_marker_names(panel_df)
        errors += cls._validate_control_column(panel_df)
        errors += cls._validate_uniprot_ids(panel_df)
        errors += cls._validate_sequences(panel_df, "sequence_1")
        errors += cls._validate_sequences(panel_df, "sequence_2")
        errors += cls._validate_hashing_marker_ids(panel_df)
        return errors

    @staticmethod
    def _validate_hashing_marker_ids(panel_df: pd.DataFrame) -> list[str]:
        """Return errors when hashing ids lack a ``-<digits>`` suffix or nest."""
        hashing_ids = _hashing_marker_ids(panel_df)
        if not hashing_ids:
            return []
        errors: list[str] = []
        missing_suffix = [
            marker_id
            for marker_id in sorted(hashing_ids)
            if split_hashing_marker_id(marker_id) is None
        ]
        if missing_suffix:
            errors.append(
                "Hashing marker ids must end with -<digits> (e.g. B2M-1). "
                f"Offending values: {missing_suffix}"
            )
        nested = sorted(
            marker_id
            for marker_id in hashing_ids
            if (base := collapsed_hashing_marker_id(marker_id)) != marker_id
            and base in hashing_ids
        )
        if nested:
            errors.append(
                "Hashing marker ids must not collapse to another hashing id "
                f"(e.g. B2M-1-1 next to B2M-1). Offending values: {nested}"
            )
        return errors

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
