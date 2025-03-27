"""Pixelator for use with MPX data.

Copyright © 2022 Pixelgen Technologies AB.
"""

# Adding imports here as shortcuts to be able to import like
# import pixelator as mpx
# mpx.read("<file path>")
# and similar
from pixelator.mpx.pixeldataset import read, PixelDataset  # noqa
from pixelator.mpx.pixeldataset.aggregation import simple_aggregate  # noqa


__all__ = ["read", "simple_aggregate", "PixelDataset"]
