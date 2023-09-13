"""
This module contains helper typehints for the pixelator package.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations

import os
from pathlib import Path, PurePath
from typing import Union

# type alias for path-like objects
PathType = Union[str, Path, PurePath, os.PathLike]
