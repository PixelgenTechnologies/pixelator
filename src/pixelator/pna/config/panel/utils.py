"""Small helpers shared by the PNA panel modules.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from pixelator.common.types import PathType

if TYPE_CHECKING:
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


def sample_hashing_mask(sample_hashing: pd.Series) -> pd.Series:
    """Normalize a ``sample_hashing`` column to a boolean mask.

    Panel CSV parsing converts ``yes``/``no`` to bool; AnnData / Polars
    round-trips may still expose strings (``yes``/``no`` or ``True``/``False``).

    Args:
        sample_hashing: Column values from a panel dataframe.

    Returns:
        Boolean series aligned to ``sample_hashing`` (``True`` = hashing marker).
    """
    if pd.api.types.is_bool_dtype(sample_hashing):
        return sample_hashing.fillna(False)
    normalized = sample_hashing.astype(str).str.strip().str.lower()
    return normalized.isin(["yes", "true", "1"])


def _resolve_panel_source_from_pxl(
    pxl_data: PNAPixelDataset,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> tuple[Optional[str], Optional[PathType]]:
    """Fill missing file_name/filepath from a single-file PNAPixelDataset."""
    if file_name is not None and filepath is not None:
        return file_name, filepath

    file_mapping = getattr(pxl_data.view, "_db_to_file_mapping", None) or {}
    if len(file_mapping) == 1:
        source_path = Path(next(iter(file_mapping.values())).path)
        if filepath is None:
            filepath = source_path
        if file_name is None:
            file_name = source_path.name
    return file_name, filepath
