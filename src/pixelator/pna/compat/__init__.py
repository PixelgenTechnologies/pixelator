"""Copyright © 2025 Pixelgen Technologies AB."""

import pandas as pd

from pixelator.pna.pixeldataset.legacy import PNALegacyPixelDataset


def polarity_score(pxl_dataset: PNALegacyPixelDataset) -> pd.DataFrame:
    """Extract polarity scores from a PNAPixelDataset."""
    return pxl_dataset._backend._datastore.read_polarization()


def colocalization_score(pxl_dataset: PNALegacyPixelDataset) -> pd.DataFrame:
    """Extract colocalization scores from a PNAPixelDataset."""
    return pxl_dataset._backend._datastore.read_colocalization()
