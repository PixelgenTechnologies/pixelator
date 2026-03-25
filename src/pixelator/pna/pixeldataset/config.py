"""Configuration objects for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PixelDatasetConfig:
    """Configuration for a PixelDataset."""

    adata_join_method: Literal["inner", "outer"] = "inner"
