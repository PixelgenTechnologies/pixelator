"""Marker panel management for different Molecular Pixelation assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import pydantic
import ruamel.yaml as yaml
from packaging.version import Version

from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.common.config import Config


class AntibodyPanelMetadata(pydantic.BaseModel):
    """Class representing the metadata of a Molecular Pixelation antibody panel."""

    model_config = pydantic.ConfigDict(extra="ignore")

    version: str
    name: str
    product: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = []
    archived: Optional[bool] = False
    panel_type: Optional[str] = None

    @pydantic.field_validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        """Validate that the panel version string is parseable.

        Args:
            v: Version string from panel metadata.

        Returns:
            The input version string when validation succeeds.

        Raises:
            packaging.version.InvalidVersion: If the value is not a valid version.

        """
        Version(v)  # will raise if not a valid version string
        return v

    @classmethod
    def from_panel_csv(cls, panel_file: PathType) -> AntibodyPanelMetadata:
        """Create an AntibodyPanelMetadata object from a panel csv file."""
        yaml_loader = yaml.YAML(typ="safe")

        metadata_lines = []
        with open(str(panel_file), "r") as f:
            for line in f:
                if line.startswith("# "):
                    metadata_lines.append(line[2:])
                else:
                    # first line after header must be csv column names
                    # metadata_lines.append(f"panel_columns: [{line.strip()}]\n")
                    break

        metadata = "".join(
            metadata_lines
            # reorder to place csv columns before last "---" to generate a valid yaml
            # metadata_lines[:-2] + [metadata_lines[-1]] + [metadata_lines[-2]]
        )
        raw_config = list(yaml_loader.load_all(metadata))

        if len(raw_config) == 0:
            raise ValueError(f"No header / metadata found in panel file {panel_file}")

        frontmatter = raw_config[0]
        print(frontmatter)
        return cls.model_validate(frontmatter)

class AntibodyPanel:
    """Class representing a Molecular Pixelation antibody panel."""

    # required columns
    _REQUIRED_COLUMNS = [
        "marker_id",
        "control",
        "nuclear",
        "sequence",
        "conj_id",
    ]

    # and these should have unique values
    _UNIQUE_COLUMNS = [
        "marker_id",
        "sequence",
    ]

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
    def from_csv(cls, filename: PathType) -> "AntibodyPanel":
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
    def validate_antibody_panel(cls, panel_df: pd.DataFrame) -> list[str]:
        """Perform validation on an antibody panel file.

        Will try to find as many issues as possible.

        This will not directly raise the issue (since that makes it difficult
        to find multiple problems at once) instead it will return a list of str
        (one for each issue).

        Usage example:
        ```
        >>> errors = panel.validate_antibody_panel(panel_df)
        ... if len(errors) > 0:
        ...     AssertionError("There was a problem with the panel data!")
        ````

        Args:
            panel_df: Panel dataframe to validate.

        Returns:
            A list of validation error messages. An empty list means valid input.

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

        return errors

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

        panel = cls._transform_legacy_panels(panel)

        # assign the sequence (unique) as index
        panel.index = panel.sequence  # type: ignore

        # return a local copy
        return panel.copy()

    @classmethod
    def _transform_legacy_panels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transform legacy panels to the new format.

        :param df: DataFrame with data of the panel to validate
        :returns pd.DataFrame: The in-place modified input dataframe
        """
        # update control and nuclear column to boolean
        TR_TABLE = {"(?i)yes": "True", "(?i)no": "False"}

        df["control"] = (
            df["control"]
            .astype("string[pyarrow]")
            .fillna("")
            .replace(TR_TABLE, regex=True)
            .astype(bool)
        )
        df["nuclear"] = (
            df["nuclear"]
            .astype("string[pyarrow]")
            .fillna("")
            .replace(TR_TABLE, regex=True)
            .astype(bool)
        )

        return df

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
            raise ValueError(f"No header / metadata found in panel file {file}")

        frontmatter = raw_config[0]
        return AntibodyPanelMetadata.model_validate(frontmatter)

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


def load_antibody_panel(config: Config, panel: PathType) -> AntibodyPanel:
    """Load an antibody panel from a file or from the config file.

    :param config: the config object
    :param panel: the path to the panel file or the name of the
        panel in the config file

    :returns: the antibody panel
    :rtype: AntibodyPanel
    """
    panel_str = str(panel)
    panel_from_config = config.get_panel(panel_str)

    if panel_from_config is not None:
        logger.info("Found panel in config file: %s", panel_from_config.name)
        return panel_from_config

    panel_obj = AntibodyPanel.from_csv(panel)
    return panel_obj
