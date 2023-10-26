"""Top-level package for Pixelator.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from importlib import metadata

__version__ = "0.0.0"

try:
    __version__ = metadata.version("pixelgen-pixelator")
except metadata.PackageNotFoundError:
    pass

# Adding imports here as shortcuts to be able to import like
# import pixelator as mpx
# mpx.read("<file path>")
# and similar
from pixelator.pixeldataset import read, PixelDataset  # noqa
from pixelator.pixeldataset.aggregation import simple_aggregate  # noqa


__all__ = ["read", "simple_aggregate", "PixelDataset"]
