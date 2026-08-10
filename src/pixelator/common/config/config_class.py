"""Classes and functions for Pixelator configuration files and assay settings.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import typing
from typing import Optional, Tuple

from pixelator.common.exceptions import PixelatorBaseException

DNA_CHARS = {"A", "C", "G", "T"}

RangeType = typing.TypeVar(
    "RangeType", Tuple[int, int], Tuple[Optional[int], Optional[int]]
)


class PanelException(PixelatorBaseException):
    """Exception raised for failures to load a panel into the global configuration."""

    pass
