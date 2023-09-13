"""Top-level package for Pixelator.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from importlib import metadata

__version__ = "0.0.0"
try:
    __version__ = metadata.version("pixelator")
except metadata.PackageNotFoundError:
    pass

# Adding imports here as shortcuts to be able to import like
# import pixelator as mpx
# mpx.read("<file path>")
# and similar
from pixelator.pixeldataset import read, simple_aggregate, PixelDataset  # noqa


__all__ = ["read", "simple_aggregate", "PixelDataset"]
