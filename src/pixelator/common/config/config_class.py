"""Classes and functions for Pixelator configuration files and assay settings.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib_resources
import semver

from pixelator.common.config.assay import Assay
from pixelator.common.exceptions import PixelatorBaseException
from pixelator.common.types import PathType

DNA_CHARS = {"A", "C", "G", "T"}

RangeType = typing.TypeVar(
    "RangeType", Tuple[int, int], Tuple[Optional[int], Optional[int]]
)


class PanelException(PixelatorBaseException):
    """Exception raised for failures to load a panel into the global configuration."""

    pass
