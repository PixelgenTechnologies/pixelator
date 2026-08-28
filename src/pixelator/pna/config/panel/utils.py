"""Small helpers shared by the PNA panel modules.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from pixelator.common.types import PathType

if TYPE_CHECKING:
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


def sample_hashing_mask(sample_hashing: pd.Series) -> pd.Series:
    """Normalize a ``sample_hashing`` column to a boolean mask.

    Panel CSV parsing converts ``yes``/``no`` to bool. Concatenating a panel
    that omits the column with one that has bool values can upcast to float
    (``True`` → ``1.0``) because missing cells become NaN. AnnData / Polars
    round-trips may still expose strings (``yes``/``no`` or ``True``/``False``).

    Args:
        sample_hashing: Column values from a panel dataframe.

    Returns:
        Boolean series aligned to ``sample_hashing`` (``True`` = hashing marker).
    """
    if pd.api.types.is_bool_dtype(sample_hashing):
        return sample_hashing.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(sample_hashing):
        return sample_hashing.fillna(0).astype(bool)
    normalized = sample_hashing.astype(str).str.strip().str.lower()
    return normalized.isin(["yes", "true", "1", "1.0"])


# Trailing ``-<digits>`` on a hashing antibody id is the hash group
# (``B2M-1`` → base ``B2M``, group ``1``). Sample calling collapses those
# ids to the base name. This pattern also matches non-hashing markers such as
# ``PD-1`` and ``TIM-3``; it must only be applied to ids already selected via
# the panel ``sample_hashing`` column.
_HASHING_MARKER_ID_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d+)$")


def split_hashing_marker_id(marker_id: str) -> tuple[str, str] | None:
    """Parse the hash-group suffix of an already-identified hashing marker id.

    ``B2M-1`` becomes ``("B2M", "1")``. Returns ``None`` when the id has no
    trailing ``-<digits>`` suffix.

    This is not a classifier. Biological names such as ``PD-1`` and
    ``TIM-3`` match the same pattern; callers must pass only marker ids
    flagged by the panel ``sample_hashing`` column.

    Args:
        marker_id: Marker id from a row with ``sample_hashing`` true.
    """
    match = _HASHING_MARKER_ID_RE.fullmatch(marker_id)
    if match is None:
        return None
    return match.group("base"), match.group("index")


def collapsed_hashing_marker_id(marker_id: str) -> str:
    """Return the post–sample-calling marker id for a hashing antibody.

    Strips a trailing ``-<digits>`` hash group (``B2M-1`` → ``B2M``). Ids
    without that suffix are returned unchanged.

    Only use this on ids from the panel ``sample_hashing`` column. Applying
    it to ``PD-1`` or ``TIM-3`` would incorrectly yield ``PD`` / ``TIM``.
    """
    parts = split_hashing_marker_id(marker_id)
    return parts[0] if parts is not None else marker_id


def nested_hashing_marker_ids(hashing_marker_ids: Iterable[str]) -> list[str]:
    """Return hashing ids whose collapsed name is itself a hashing id.

    ``B2M-1-1`` collapses to ``B2M-1``. That is invalid when ``B2M-1`` is
    also a hashing marker.
    """
    ids = {str(marker_id) for marker_id in hashing_marker_ids}
    return sorted(
        hid
        for hid in ids
        if (base := collapsed_hashing_marker_id(hid)) != hid and base in ids
    )


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
