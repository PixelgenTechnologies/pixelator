"""Marker panel management for different PNA assays.

A panel describes the antibodies (markers) present in a sample, together with
the clone sequences used to identify them. The implementation is split over a
few modules:

- `base` defines `PNAPanel`, the read interface shared by every panel type
  (marker helpers, size, Polars conversion, schema validation).
- `partial` defines `PartialPNAAntibodyPanel` — one panel loaded from a single
  CSV — and the typed variants `PNABasePanel`, `PNAAddonPanel` and
  `PNASampleHashingPanel` that bind a fixed `PanelType`.
- `dispatch` maps panel metadata to the concrete class to instantiate, and
  loads a CSV into whichever class the file header asks for.
- `combination` defines `PNAAntibodyPanelCombination`, the concatenation of the
  panels used together in one sample (base + hashing + addon).
- `diff` defines `PNAAntibodyPanelDiff`, which compares two panels by clone
  sequence and can apply patch-level panel upgrades to AnnData.
- `loaders` builds panels from AnnData, pixel datasets, and the pixelator
  config.

Everything that makes up the public interface is re-exported here, so
``from pixelator.pna.config.panel import ...`` keeps working regardless of
which module a name lives in.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType

from .base import PNAPanel
from .combination import PNAAntibodyPanelCombination
from .diff import PNAAntibodyPanelDiff
from .dispatch import get_panel_type_from_metadata, panel_from_csv
from .loaders import load_antibody_panel, panel_from_adata, panel_from_pxl_dataset
from .partial import (
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNABasePanel,
    PNASampleHashingPanel,
)
from .utils import sample_hashing_mask

if TYPE_CHECKING:
    # Deprecated alias of :class:`PartialPNAAntibodyPanel` (warns on runtime access).
    PNAAntibodyPanel = PartialPNAAntibodyPanel

__all__ = [
    "AntibodyPanelMetadata",
    "PanelType",
    "PNAPanel",
    "PartialPNAAntibodyPanel",
    "PNAAntibodyPanel",
    "PNABasePanel",
    "PNAAddonPanel",
    "PNASampleHashingPanel",
    "PNAAntibodyPanelCombination",
    "PNAAntibodyPanelDiff",
    "get_panel_type_from_metadata",
    "load_antibody_panel",
    "panel_from_adata",
    "panel_from_csv",
    "panel_from_pxl_dataset",
    "sample_hashing_mask",
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
        # Cache so repeated access does not re-warn.
        globals()["PNAAntibodyPanel"] = PartialPNAAntibodyPanel
        return PartialPNAAntibodyPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
