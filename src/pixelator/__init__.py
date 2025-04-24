"""Top-level package for Pixelator.

Copyright © 2022 Pixelgen Technologies AB.
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
from pixelator.mpx.pixeldataset import PixelDataset as MPXPixelDataset  # noqa
from pixelator.mpx.pixeldataset import read as read_mpx
from pixelator.mpx.pixeldataset.aggregation import (
    simple_aggregate as simple_aggregate_mpx,
)  # noqa

from pixelator.pna import read as read_pna

__all__ = ["read_mpx", "simple_aggregate_mpx", "MPXPixelDataset", "read_pna"]
