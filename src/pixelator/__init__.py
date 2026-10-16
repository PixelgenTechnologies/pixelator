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
# import pixelator.pna as pna
# pna.read("<file path>")
# and similar

from pixelator.pna import DownloadableDatasets
from pixelator.pna import read as read_pna

__all__ = [
    "read_pna",
    "DownloadableDatasets",
]
