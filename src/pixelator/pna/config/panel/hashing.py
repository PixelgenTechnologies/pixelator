"""Helpers for sample-hashing marker ids.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re

import pandas as pd

# Trailing ``-<digits>`` is the hash group (``B2M-1`` → ``B2M``). The same
# pattern matches ordinary names such as ``PD-1``, so it is only applied to
# rows already flagged by ``sample_hashing``.
_HASHING_MARKER_ID_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d+)$")


def sample_hashing_mask(sample_hashing: pd.Series) -> pd.Series:
    """Return a boolean mask for values that flag a hashing marker."""
    if pd.api.types.is_bool_dtype(sample_hashing):
        return sample_hashing.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(sample_hashing):
        return sample_hashing.fillna(0).astype(bool)
    normalized = sample_hashing.astype(str).str.strip().str.lower()
    return normalized.isin(["yes", "true"])


def split_hashing_marker_id(marker_id: str) -> tuple[str, str] | None:
    """Return ``(base, index)`` for a hashing id such as ``B2M-1``."""
    match = _HASHING_MARKER_ID_RE.fullmatch(str(marker_id))
    if match is None:
        return None
    return match.group("base"), match.group("index")


def collapsed_hashing_marker_id(marker_id: str) -> str:
    """Return the marker id sample calling stores for a hashing antibody."""
    parts = split_hashing_marker_id(marker_id)
    return parts[0] if parts is not None else str(marker_id)


def _hashing_marker_ids(panel_df: pd.DataFrame) -> set[str]:
    """Return hashing marker ids, or an empty set when the column is absent."""
    if "sample_hashing" not in panel_df.columns:
        return set()
    mask = sample_hashing_mask(panel_df["sample_hashing"])
    return {str(marker_id) for marker_id in panel_df.index[mask]}
