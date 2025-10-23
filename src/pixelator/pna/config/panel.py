"""Marker panel management for different Molecular Pixelation assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


import re
from typing import TYPE_CHECKING, Optional

import pandas as pd
import ruamel.yaml as yaml

from pixelator import read_pna
from pixelator.common.config import AntibodyPanelMetadata
from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig


class PNAAntibodyPanel:
    """Class representing a Molecular Pixelation antibody panel."""

    # required columns
    _REQUIRED_COLUMNS = [
        "marker_id",
        "control",
        "sequence_1",
        "sequence_2",
        "conj_id",
    ]

    # and these should have unique values
    _UNIQUE_COLUMNS = ["sequence_1", "sequence_2", "conj_id"]

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata,
        file_name: Optional[str] = None,
    ) -> None:
        """Load a panel from a dataframe and metadata.

        :param df: The dataframe containing the panel information.
        :param metadata: The metadata for the panel.
        :param file_name: The optional basename of the file from which
            the panel is loaded.

        :returns: None
        :raises AssertionError: exception if panel file is missing,
                                invalid or with incorrect format
        """
        self._filename = file_name
        self._metadata = metadata
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

        :param filename: The path to the panel file.
        :returns: The AntibodyPanel object.
        :raises AssertionError: exception if panel file is missing,
        :rtype: AntibodyPanel
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
    def from_pxl(cls, filename: PathType) -> Self:
        """Create an AntibodyPanel from a pxl file.

        :param filename: The path to the panel file.
        :returns: The AntibodyPanel object.
        :raises AssertionError: exception if panel file is missing,
        :rtype: AntibodyPanel
        """
        pxl_file = Path(filename)

        if not pxl_file.is_file() or pxl_file.suffix != ".pxl":
            raise AssertionError(
                f"Panel file {filename} not found or has an incorrect format"
            )

        logger.debug("Creating Antibody panel from file %s", filename)

        pxl_data = read_pna(pxl_file)
        adata = pxl_data.adata()
        panel_metadata = adata.uns["panel_metadata"]
        panel_columns = panel_metadata.pop("panel_columns")
        df = adata.var.reset_index(names="marker_id")[panel_columns]
        metadata = AntibodyPanelMetadata.model_validate(panel_metadata)

        logger.debug("Antibody panel from file %s created", filename)

        return cls(df, metadata, file_name=pxl_file.name)

    @property
    def name(self) -> Optional[str]:
        """The name defined in the panel metadata."""
        return self._metadata.name

    @property
    def version(self) -> Optional[str]:
        """Return the panel file version."""
        return self._metadata.version

    @property
    def description(self) -> Optional[str]:
        """Return the panel file description."""
        return self._metadata.description

    @property
    def aliases(self) -> list[str]:
        """Return the (optional) list of panel file aliases."""
        return self._metadata.aliases

    @classmethod
    def _parse_header(cls, file: Path) -> AntibodyPanelMetadata:
        """Parse front-matter YAML from a file.

        :param file: the file to parse
        :return AntibodyPanelMetadata: a pydantic model with the metadata
        """
        yaml_loader = yaml.YAML(typ="safe")

        metadata_lines = []
        with open(str(file), "r") as f:
            for line in f:
                if line.startswith("# "):
                    metadata_lines.append(line[2:])
                else:
                    break

        metadata = "".join(metadata_lines)
        raw_config = list(yaml_loader.load_all(metadata))

        if len(raw_config) == 0:
            warnings.warn(f"Expected a YAML frontmatter in {file}")
            return AntibodyPanelMetadata(version=None, name=None, description=None)

        frontmatter = raw_config[0]
        return AntibodyPanelMetadata.model_validate(frontmatter)

    @classmethod
    def _parse_panel(cls, panel_file: Path) -> pd.DataFrame:
        panel = pd.read_csv(str(panel_file), comment="#")

        # validate the panel
        errors = cls.validate_antibody_panel(panel)
        if len(errors) > 0:
            msg_str = "\n".join(errors)
            raise AssertionError(
                f"The following errors were found validating the panel: {msg_str}"
            )

        # assign the sequence (unique) as index
        panel.index = panel.conj_id  # type: ignore

        # return a local copy
        return panel.copy()

    @cached_property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self._df[self._df["control"]].marker_id.unique())

    @cached_property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self._df.marker_id.unique())

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
        return self._df.shape[0]

    def get_marker_id(self, seq: str) -> str:
        """Return the marker name."""
        return self._df.loc[seq].marker_id

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

    @classmethod
    def validate_antibody_panel(cls, panel_df: pd.DataFrame) -> list[str]:
        """Validate the antibody panel dataframe.

        :param panel_df: The dataframe containing the panel information.
        :returns: A list of errors found in the panel.
        """
        errors = []

        # some basic sanity check on the panel size and columns
        if not set(cls._REQUIRED_COLUMNS).issubset(set(panel_df.columns)):
            missing_columns = set(cls._REQUIRED_COLUMNS) - set(panel_df.columns)
            errors.append(f"Panel has missing required columns: {missing_columns}")
            return errors

        if panel_df.shape[0] == 0:
            errors.append("Panel file is empty")
            return errors

        # sanity check on the unique columns
        for col in cls._UNIQUE_COLUMNS:
            if not len(panel_df[col].unique()) == len(panel_df[col]):
                errors.append(f"All values in column: {col} were not unique")

        if any(panel_df["marker_id"].str.contains("_")):
            # Markers should not contain underscores since this messes
            # things up with Seurat on the R side
            errors.append(
                "The marker_id column should not contain underscores. "
                "Please use dashes instead. Offending values: "
                f"{panel_df['marker_id'][panel_df['marker_id'].str.contains('_')]}"
            )

        # Check UniProt IDs format conforming to the UniProt naming convention. Empty IDs are allowed.
        if "uniprot_id" in panel_df.columns:
            # Pattern for valid UniProt IDs
            pattern = r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"

            def check_id(id_str):
                if pd.isna(id_str):
                    return True
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


def load_antibody_panel(config: PNAConfig, panel: PathType) -> PNAAntibodyPanel:
    """Load an antibody panel from a file or from the config file.

    :param config: the config object
    :param panel: the path to the panel file or the name of the
        panel in the config file

    :returns: the antibody panel
    :rtype: PNAAntibodyPanel
    """
    panel_str = str(panel)
    panel_from_config = config.get_panel(panel_str)

    if panel_from_config is not None:
        logger.info("Found panel in config file: %s", panel_from_config.name)
        return panel_from_config

    panel_obj = PNAAntibodyPanel.from_csv(panel)
    return panel_obj
