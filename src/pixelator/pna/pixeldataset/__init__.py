"""PNA Pixel Dataset.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path

from pixelator.pna.pixeldataset.config import PixelDatasetConfig
from pixelator.pna.pixeldataset.dataset import PNAPixelDataset
from pixelator.pna.pixeldataset.edgelist import Edgelist
from pixelator.pna.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pna.pixeldataset.proximity import Proximity
from pixelator.pna.pixeldataset.saver import PixelDatasetSaver
from pixelator.pna.pixeldataset.types import Component


def read(paths: Path | list[Path] | str | list[str]) -> PNAPixelDataset:
    """Read a PNAPixelDataset from one or more provided .pxl file(s).

    :param path: path to the file to read
    :return: an instance of `PNAPixelDataset`
    """
    if not paths:
        raise ValueError(
            "No paths provided to read function, did you pass an empty list?"
        )
    if not isinstance(paths, list):
        paths = [paths]  # type: ignore
    normalized_paths = [Path(p) for p in paths]  # type: ignore
    return PNAPixelDataset.from_pxl_files(normalized_paths)


__all__ = [
    "Component",
    "Edgelist",
    "PixelDatasetConfig",
    "PixelDatasetSaver",
    "PNAPixelDataset",
    "PreComputedLayouts",
    "Proximity",
]
