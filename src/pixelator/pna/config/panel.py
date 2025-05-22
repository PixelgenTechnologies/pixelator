"""Marker panel management for different Molecular Pixelation assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from typing import TYPE_CHECKING, Optional

import pandas as pd

from pixelator.common.config import AntibodyPanel
from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.config.config_class import AntibodyPanelMetadata
    from pixelator.pna.config.config_class import PNAConfig


class PNAAntibodyPanel(AntibodyPanel):
    """Class representing a Molecular Pixelation antibody panel."""

    # required columns
    _REQUIRED_COLUMNS = [
        "marker_id",
        "control",
        "nuclear",
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

    @classmethod
    def validate_antibody_panel(cls, panel_df: pd.DataFrame) -> list[str]:
        """Validate the antibody panel dataframe.

        :param panel_df: The dataframe containing the panel information.
        :returns: A list of errors found in the panel.
        """
        errors = super().validate_antibody_panel(panel_df)

        if any(panel_df["marker_id"].str.contains("_")):
            # Markers should not contain underscores since this messes
            # things up with Seurat on the R side
            errors.append(
                "The marker_id column should not contain underscores. "
                "Please use dashes instead. Offending values: "
                f"{panel_df['marker_id'][panel_df['marker_id'].str.contains('_')]}"
            )

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
