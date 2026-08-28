"""Configuration models and loaders for PNA assays and panels.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType
from pixelator.common.config.utils import load_yaml_file
from pixelator.pna.config.assay import (
    AssayModel,
    PNAAssay,
    PNARegionType,
    Region,
    RegionModel,
    SequenceType,
    get_position_in_parent,
)
from pixelator.pna.config.config_class import load_assays_package
from pixelator.pna.config.config_instance import pna_config
from pixelator.pna.config.panel import (
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNAAntibodyPanelCombination,
    PNABasePanel,
    PNAPanel,
    PNASampleHashingPanel,
    load_antibody_panel,
    panel_from_adata,
    panel_from_csv,
    panel_from_pxl_dataset,
)

if TYPE_CHECKING:
    # Deprecated alias; resolved at runtime via ``__getattr__`` with a warning.
    from pixelator.pna.config.panel import PNAAntibodyPanel as PNAAntibodyPanel

__all__ = [
    "AssayModel",
    "RegionModel",
    "PNARegionType",
    "SequenceType",
    "PNAAssay",
    "Region",
    "get_position_in_parent",
    "pna_config",
    "load_assays_package",
    "load_yaml_file",
    "PanelType",
    "PNAPanel",
    "PartialPNAAntibodyPanel",
    "PNAAntibodyPanel",
    "PNABasePanel",
    "PNAAddonPanel",
    "PNASampleHashingPanel",
    "PNAAntibodyPanelCombination",
    "AntibodyPanelMetadata",
    "load_antibody_panel",
    "panel_from_csv",
    "panel_from_adata",
    "panel_from_pxl_dataset",
]


def __getattr__(name: str):
    """Resolve the deprecated ``PNAAntibodyPanel`` alias with a warning."""
    if name == "PNAAntibodyPanel":
        warnings.warn(
            "PNAAntibodyPanel is deprecated and will be removed in a future release. "
            "Use PartialPNAAntibodyPanel (or a typed subclass), or "
            "PNAAntibodyPanelCombination for multi-panel samples.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Cache so repeated access does not re-warn; keep isinstance compat.
        globals()["PNAAntibodyPanel"] = PartialPNAAntibodyPanel
        return PartialPNAAntibodyPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
