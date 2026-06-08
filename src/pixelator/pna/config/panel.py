"""Marker panel management for different PNA assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import re
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl

from pixelator.common.config.panel import (
    AntibodyPanelMetadata,
    PanelType,
    parse_panel_header_metadata,
)
from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


class PartialPNAAntibodyPanel:
    """Class representing a PNA antibody panel."""

    _panel_type: PanelType = PanelType.PARTIAL

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
    ) -> None:
        """Load a panel from a dataframe and metadata.

        Args:
            df: The dataframe containing the panel information.
            metadata: The metadata for the panel.
            file_name: The optional basename of the file from which the panel is loaded.

        Returns:
            None
        Raises:
            AssertionError: exception if panel file is missing, invalid or with incorrect format
        """
        self._filename = file_name

        if metadata is None:
            raise ValueError("Panel metadata cannot be None")
        if (
            self.__class__._panel_type is PanelType.PARTIAL
            and metadata.panel_type is None
        ):
            # if panel type is missing from metadata use partial as default legacy behavior
            metadata.panel_type = PanelType.PARTIAL
        elif metadata.panel_type != self.__class__._panel_type:
            raise ValueError(
                f"Panel metadata panel_type {metadata.panel_type!r} does not match "
                + f"{self.__class__.__name__} (expected {self.__class__._panel_type.value})."
            )
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

        return cls(df, metadata, file_name=panel_file.name)

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
    def from_adata(cls, adata: AnnData, file_name: Optional[str] = None):
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
                "The provided AnnData object does not contain panel columns information in the "
                + "metadata. Please, regenerate your data with the most recent version of "
                + "pixelator."
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
        return AntibodyPanelMetadata.from_panel_csv(file)

    @classmethod
    def _parse_panel(cls, panel_file: Path) -> pd.DataFrame:
        panel = pd.read_csv(str(panel_file), comment="#", index_col="marker_id").fillna(
            ""
        )

        panel["control"] = panel["control"].map(lambda s: s.lower() == "yes")
        if "sample_hashing" in panel.columns:
            panel["sample_hashing"] = panel["sample_hashing"].map(
                lambda s: s.lower() == "yes"
            )

        return panel.copy()

    @cached_property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self.df[self.df["control"]].index)

    @cached_property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self.df.index.unique())

    @property
    def df(self) -> pd.DataFrame:
        """Return the panel dataframe."""
        return self._df

    @property
    def filename(self) -> Optional[str]:
        """Return the filename of the marker panel."""
        return self._filename

    @cached_property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self.df.shape[0]

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

    def to_polars(self) -> pl.DataFrame:
        """Convert the panel to a Polars DataFrame."""
        return pl.from_pandas(self.df, include_index=True)

    def __eq__(self, other: object) -> bool:
        """Check if two panels are equal based on their dataframes and metadata.

        Args:
            other: Panel to compare for equality.
        """
        if not isinstance(other, PartialPNAAntibodyPanel):
            raise ValueError("Can only compare with another PartialPNAAntibodyPanel")
        return self.df.equals(other.df) and self.metadata == other.metadata


def get_panel_type_from_metadata(
    metadata: AntibodyPanelMetadata,
) -> type[PartialPNAAntibodyPanel]:
    """Get the panel class type from the panel metadata."""
    match metadata.panel_type:
        case PanelType.PARTIAL:
            return PartialPNAAntibodyPanel
        case PanelType.BASE:
            return PNABasePanel
        case PanelType.ADDON:
            return PNAAddonPanel
        case PanelType.SAMPLE_HASHING:
            return PNASampleHashingPanel
        case None:
            warnings.warn(
                "Panel metadata has no panel_type. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
                UserWarning,
            )
            return PartialPNAAntibodyPanel
        case _:
            warnings.warn(
                f"Unknown panel type {metadata.panel_type} in panel metadata. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
                UserWarning,
            )
            return PartialPNAAntibodyPanel


def panel_from_adata(adata: AnnData) -> PartialPNAAntibodyPanel:
    """Create a PartialPNAAntibodyPanel from an AnnData object."""
    if "num_partial_panels" in adata.uns:
        return PNAAntibodyPanelCombination.from_adata(adata)
    panel_type = get_panel_type_from_metadata(
        AntibodyPanelMetadata.from_adata(adata)[0]
    )
    return panel_type.from_adata(adata)


def panel_from_pxl_dataset(
    pxl_data: PNAPixelDataset,
) -> PartialPNAAntibodyPanel | PNAAntibodyPanelCombination:
    """Create a PartialPNAAntibodyPanel or PNAAntibodyPanelCombination from a PNAPixelDataset object."""
    return panel_from_adata(pxl_data.adata())


def panel_from_csv(panel_file: PathType) -> PartialPNAAntibodyPanel:
    """Create a PartialPNAAntibodyPanel from a csv panel file."""
    panel_file = Path(panel_file)
    if not panel_file.is_file() or panel_file.suffix != ".csv":
        raise AssertionError(
            f"Panel file {panel_file} not found or has an incorrect format"
        )

    metadata = AntibodyPanelMetadata.from_panel_csv(panel_file)
    panel_type = get_panel_type_from_metadata(metadata)
    return panel_type.from_csv(panel_file)


def load_antibody_panel(
    config: PNAConfig, requested_panels: PathType | list[PathType] | list[str]
) -> PNAAntibodyPanelCombination:
    """Load an antibody panel from a file or from the config file.

    Args:
        config: the config object.
        requested_panels: the path to the panel file(s) or the name(s) of the
            panel(s) in the config file.

    Returns:
        The loaded antibody panel as a PNAAntibodyPanelCombination.
    """
    return_panels = []
    for panel in (
        requested_panels if isinstance(requested_panels, list) else [requested_panels]
    ):
        panel_str = str(panel)
        logger.debug("Loading panel %s", panel_str)
        panel_from_config = config.get_panel(panel_str)

        if panel_from_config is not None:
            logger.info("Found panel in config file: %s", panel_from_config.name)
            return_panels.append(panel_from_config)
            continue

        panel_obj = panel_from_csv(panel)
        logger.info(
            "Loaded %s %s from CSV file: %s",
            panel_obj.__class__.__name__,
            panel_obj.name,
            panel_obj.filename,
        )
        return_panels.append(panel_obj)
    return PNAAntibodyPanelCombination.from_list_of_subpanels(return_panels)


class PNAAntibodyPanelDiff:
    """Class representing the differences between two PNAAntibodyPanel objects."""

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
        for idx in range(adata.uns.get("num_partial_panels", 0)):
            metadata_key = f"panel_metadata__{idx}"
            metadata = AntibodyPanelMetadata.model_validate(adata.uns[metadata_key])
            if (
                metadata.name == self.panel_1.name
                and metadata.version == self.panel_1.version
            ):
                adata.uns[metadata_key] = self.panel_2.metadata.to_dict()
                adata.uns[f"panel_df__{idx}"] = self.panel_2.df.to_csv()
                break

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


class PNAAntibodyPanelCombination(PartialPNAAntibodyPanel):
    """Class representing a combination of PNA antibody panels used in a sample.

    This can be a combination of base panels, sample hashing panels and addon panels.
    This represent the concat of the panels i.e. all the antibodies added to the same tube.

    Should raise loud errors if the panels are not compatible (e.g. different sequences for the same
      marker or same sequences is present in multiple panels).
    """

    _REQUIRED_COLUMNS = {
        **PartialPNAAntibodyPanel._REQUIRED_COLUMNS,
        "partial_panel_name": str,
        "partial_panel_type": str,
    }

    base_panels: list[PNABasePanel | PartialPNAAntibodyPanel]
    hashing_panels: Optional[list[PNASampleHashingPanel]]
    addon_panels: Optional[list[PNAAddonPanel]]

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata | list[AntibodyPanelMetadata],
        file_name: Optional[str] = None,
    ):
        """Initialize the PNAAntibodyPanelCombination object."""
        if metadata is None:
            raise ValueError("Panel metadata cannot be None")

        self.base_panels = []
        self.hashing_panels = None
        self.addon_panels = None

        # a combination never has a filename itself but the partial panel member might have
        self._filename = None

        if (
            "partial_panel_name" not in df.columns
            and "partial_panel_type" not in df.columns
            and isinstance(metadata, AntibodyPanelMetadata)
        ):
            panel_type_class = get_panel_type_from_metadata(metadata)
            panel = panel_type_class(df, metadata, file_name=file_name)
            self.add_panel(panel)
            self._df = self.df  # add in the extra columns
        elif (
            "partial_panel_name" in df.columns
            and "partial_panel_type" in df.columns
            and isinstance(metadata, list)
            and all(isinstance(m, AntibodyPanelMetadata) for m in metadata)
        ):
            raise NotImplementedError(
                "Please initialise using from_list_of_subpanels instead"
            )
        else:
            raise ValueError(
                "Cannot initialise PNAAntibodyPanelCombination with the provided dataframe and "
                "metadata."
            )
        self._df = self.df

    @classmethod
    def from_panel(
        cls, panel: PartialPNAAntibodyPanel
    ) -> "PNAAntibodyPanelCombination":
        """Initialize a panel combination from a single panel."""
        return cls(
            df=panel.df,
            metadata=panel.metadata,
            file_name=panel.filename,
        )

    @property  # type: ignore[override]
    def metadata(self) -> list[AntibodyPanelMetadata]:  # type: ignore[override]
        """Return the metadata for all the panels that are part of the combination."""
        return [p.metadata for p in self.partial_panels()]

    @metadata.setter  # type: ignore[override]
    def metadata(self, _value: list[AntibodyPanelMetadata]):  # type: ignore[override]
        """Set the metadata for all the panels that are part of the combination."""
        raise AttributeError("Metadata for combination panels is read-only.")

    def partial_panels(self):
        """Return a list of all the partial panels that are part of the combination."""
        return (
            self.base_panels + (self.hashing_panels or []) + (self.addon_panels or [])
        )

    @property
    def num_partial_panels(self):
        """Return the number of all the partial panels that are part of the combination."""
        return sum(1 for _ in self.partial_panels())

    @staticmethod
    def _validate_no_conflicting_duplicate_markers(df_list: list[pd.DataFrame]) -> None:
        """Reject the same marker_id with incompatible rows across partial panels."""
        seen_markers: dict[str, pd.Series] = {}
        for partial_df in df_list:
            for marker_id, row in partial_df.iterrows():
                if marker_id in seen_markers:
                    if not seen_markers[marker_id].equals(row):
                        raise ValueError(
                            "Conflicting duplicate marker_id "
                            + f"'{marker_id}' across panels in combination."
                        )
                else:
                    seen_markers[marker_id] = row

    @property
    def df(self):
        """Return the panel dataframe for the combination of panels."""
        partial_dfs = [panel.df for panel in self.partial_panels()]
        if len(partial_dfs) > 1:
            self._validate_no_conflicting_duplicate_markers(partial_dfs)

        df_list = [
            panel.to_polars()
            .with_columns(
                pl.lit(panel.name).alias("partial_panel_name"),
                pl.lit(panel.metadata.panel_type or PanelType.PARTIAL).alias(
                    "partial_panel_type"
                ),
            )
            .to_pandas()
            .set_index(self._INDEX_COLUMN)
            for panel in self.partial_panels()
        ]
        df = (
            pd.concat(
                df_list,
                axis=0,
            )
            if len(df_list) > 1
            else df_list[0]
        )
        # make sure clone sequences are unique! Otherwise raise an error
        if df.duplicated(subset=["sequence_1", "sequence_2"]).any():
            raise ValueError("Duplicate sequences found in the panel combination.")
        return df

    def add_base_panel(self, base_panel: PNABasePanel | PartialPNAAntibodyPanel):
        """Add a base panel."""
        if isinstance(base_panel, PartialPNAAntibodyPanel):
            logger.warning(
                "Adding a PartialPNAAntibodyPanel as a base panel. "
                + "this is expected legacy behavior for panels without a defined type."
                + " Consider updating the panel metadata to set the panel type."
            )
        self.base_panels.append(base_panel)
        self._df = self.df

    def add_addon_panel(self, addon_panel: PNAAddonPanel):
        """Add an addon panel."""
        if self.addon_panels is None:
            self.addon_panels = []
        self.addon_panels.append(addon_panel)
        self._df = self.df

    def add_hashing_panel(self, hashing_panel: PNASampleHashingPanel):
        """Add a sample hashing panel."""
        if self.hashing_panels is None:
            self.hashing_panels = []
        self.hashing_panels.append(hashing_panel)
        self._df = self.df

    def add_panel(
        self,
        panel: PartialPNAAntibodyPanel
        | PNABasePanel
        | PNASampleHashingPanel
        | PNAAddonPanel,
    ):
        """Add another panel to the combination."""
        # Match subclasses before PartialPNAAntibodyPanel; class patterns also
        # match instances of subclasses.
        match panel:
            case PNASampleHashingPanel():
                self.add_hashing_panel(panel)
            case PNAAddonPanel():
                self.add_addon_panel(panel)
            case PNABasePanel():
                self.add_base_panel(panel)
            case PartialPNAAntibodyPanel():
                self.add_base_panel(panel)
            case _:
                raise ValueError(f"Unknown panel type: {panel.__class__.__name__}")
        self._df = self.df

    @classmethod
    def from_list_of_subpanels(
        cls,
        panels: list[
            PartialPNAAntibodyPanel
            | PNABasePanel
            | PNASampleHashingPanel
            | PNAAddonPanel
        ],
    ) -> "PNAAntibodyPanelCombination":
        """Create a panel combination from a list of subpanels."""
        if not panels:
            raise ValueError("At least one panel is required to build a combination.")
        combination = cls.from_panel(panels[0])
        for panel in panels[1:]:
            combination.add_panel(panel)
        return combination

    @property
    def name(self) -> str:
        """Panel name from metadata.

        Returns:
            The panel name.

        """
        return " + ".join(p.metadata.name for p in self.partial_panels())

    @property
    def product(self) -> Optional[str]:
        """Product identifier from metadata, if present.

        Returns:
            Product name, or None when not provided in panel metadata.

        """
        # ok to drop None values here since we anyway use the partial panels when checking this
        # property for e.g. bumping panel versions, i.e. this string is only for display
        products = [
            p.metadata.product
            for p in self.partial_panels()
            if p.metadata.product is not None
        ]
        if not products:
            return None
        return " + ".join(products)

    @property
    def version(self) -> str:
        """Panel version from metadata.

        Returns:
            Semantic version string for this panel.

        """
        return " + ".join(p.metadata.version for p in self.partial_panels())

    @property
    def description(self) -> Optional[str]:
        """Return the panel file description."""
        return " + ".join(str(p.metadata.description) for p in self.partial_panels())

    @property
    def aliases(self) -> list[str]:
        """Return the (optional) list of panel file aliases."""
        if self.num_partial_panels == 1:
            return self.partial_panels()[0].aliases
        else:
            raise AttributeError(
                "Cannot get aliases for a combination of panels. "
                + "Aliases are only defined for individual panels."
            )

    @property
    def archived(self) -> Optional[bool]:
        """Return whether the panel is marked as archived."""
        return any(p.metadata.archived for p in self.partial_panels())

    @property
    def filename(self) -> Optional[str]:
        """Return the filename of the panel, if available."""
        if self.num_partial_panels == 1:
            return self.partial_panels()[0].filename
        else:
            raise AttributeError(
                "Cannot get filename for a combination of panels. "
                + "Filename is only defined for individual panels."
            )

    @classmethod
    def from_adata(cls, adata, file_name=None):
        """Create a panel combination from an AnnData object."""
        if "num_partial_panels" not in adata.uns:
            subpanels = [PartialPNAAntibodyPanel.from_adata(adata, file_name)]
        else:
            list_of_metadatas = AntibodyPanelMetadata.from_adata(adata)
            logger.debug(
                "Found %d panel metadata entries in the AnnData object.",
                len(list_of_metadatas),
            )
            subpanels = []
            for idx, metadata in enumerate(list_of_metadatas):
                panel_df_key = f"panel_df__{idx}"
                if panel_df_key not in adata.uns:
                    raise KeyError(
                        "The provided AnnData object contains partial panel information but is "
                        + f"missing the panel dataframe for panel at index {idx}."
                    )
                df = (
                    pd.read_csv(StringIO(adata.uns[panel_df_key]))
                    .set_index(cls._INDEX_COLUMN)
                    .fillna("")
                )
                panel_type_class = get_panel_type_from_metadata(metadata)
                subpanels.append(panel_type_class(df, metadata))

        return PNAAntibodyPanelCombination.from_list_of_subpanels(subpanels)


class PNABasePanel(PartialPNAAntibodyPanel):
    """Class representing a base panel for PNA."""

    _panel_type = PanelType.BASE


class PNAAddonPanel(PartialPNAAntibodyPanel):
    """Class representing an addon panel for PNA."""

    _panel_type = PanelType.ADDON


class PNASampleHashingPanel(PartialPNAAntibodyPanel):
    """Class representing a sample hashing panel for PNA."""

    _panel_type = PanelType.SAMPLE_HASHING

    _REQUIRED_COLUMNS = {
        **PartialPNAAntibodyPanel._REQUIRED_COLUMNS,
        "sample_hashing": bool,
    }

    @classmethod
    def validate_antibody_panel(cls, panel_df, validate_types=True):
        """Validate that the panel dataframe is a valid sample hashing panel."""
        return super().validate_antibody_panel(panel_df, validate_types) + (
            []
            if (panel_df["sample_hashing"]).all()
            else [
                "All entries in `sample_hashing` column must be 'yes' (True) for a sample hashing panel"
            ]
        )
