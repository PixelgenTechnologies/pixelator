"""Collapse UMI regions independently.

Copyright © 2025 Pixelgen Technologies AB
"""

from .collapser import (
    MarkerCorrectionStats,
    RegionCollapser,
    SingleUMICollapseSampleReport,
)
from .report import IndependentCollapseSampleReport

__all__ = [
    "RegionCollapser",
    "SingleUMICollapseSampleReport",
    "MarkerCorrectionStats",
    "IndependentCollapseSampleReport",
]
