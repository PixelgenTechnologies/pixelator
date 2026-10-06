"""Patches that make older ``.pxl`` files look like the current schema.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pixelator.pna.pixeldataset.legacy.marker_columns import LegacyMarkerQueryAdapter

__all__ = ["LegacyMarkerQueryAdapter"]
