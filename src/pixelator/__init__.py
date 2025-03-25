"""Top-level package for Pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

from importlib import metadata

__version__ = "0.0.0"

try:
    __version__ = metadata.version("pixelgen-pixelator")
except metadata.PackageNotFoundError:
    pass

# TODO Finish this!

# Adding imports here as shortcuts to be able to import like
# import pixelator as mpx
# mpx.read("<file path>")
# and similar
from pixelator_mpx.pixeldataset import PixelDataset as MPXPixelDataset  # noqa
from pixelator_mpx.pixeldataset import read as read_mpx
from pixelator_mpx.pixeldataset.aggregation import (
    simple_aggregate as mpx_simple_aggregate,
)  # noqa

__all__ = ["read_mpx", "mpx_simple_aggregate", "MPXPixelDataset"]
