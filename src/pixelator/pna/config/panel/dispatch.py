"""Resolve the concrete PNA panel class to use from panel metadata.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.partial import (
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNABasePanel,
    PNASampleHashingPanel,
)


def get_panel_type_from_metadata(
    metadata: AntibodyPanelMetadata,
) -> type[PartialPNAAntibodyPanel]:
    """Map panel metadata to the concrete panel class to instantiate.

    Args:
        metadata: Panel metadata, typically from CSV front-matter.

    Returns:
        :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`,
        :class:`~pixelator.pna.config.panel.partial.PNABasePanel`,
        :class:`~pixelator.pna.config.panel.partial.PNAAddonPanel`, or
        :class:`~pixelator.pna.config.panel.partial.PNASampleHashingPanel`.
        Missing or unknown ``panel_type`` values fall back to
        :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`
        (logged at debug).
    """
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
            logger.debug(
                "Panel metadata has no panel_type. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
            )
            return PartialPNAAntibodyPanel
        case _:
            logger.debug(
                "Unknown panel type %s in panel metadata. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
                metadata.panel_type,
            )
            return PartialPNAAntibodyPanel


def panel_from_csv(panel_file: PathType) -> PartialPNAAntibodyPanel:
    """Create a typed panel from a CSV file, dispatching on metadata ``panel_type``.

    Prefer this over
    :meth:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel.from_csv`
    when the concrete class should follow the file header.

    Args:
        panel_file: Path to a ``.csv`` panel file with YAML front-matter.

    Returns:
        An instance of
        :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel` or
        a typed subclass.

    Raises:
        AssertionError: If the file is missing or not a ``.csv``.
    """
    panel_file = Path(panel_file)
    if not panel_file.is_file() or panel_file.suffix != ".csv":
        raise AssertionError(
            f"Panel file {panel_file} not found or has an incorrect format"
        )

    metadata = AntibodyPanelMetadata.from_panel_csv(panel_file)
    panel_type = get_panel_type_from_metadata(metadata)
    return panel_type.from_csv(panel_file)
