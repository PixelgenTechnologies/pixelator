"""AnnData ``uns`` keys for sample-calling layout.

Panel upgrade code imports this module; it must not import
:mod:`pixelator.pna.anndata` or :mod:`pixelator.pna.sample_calling` (both load
panels and would cycle through :mod:`pixelator.pna.config.panel.diff`).

Copyright © 2025 Pixelgen Technologies AB.
"""

# This is not a great place for these functions they should probably be placed under
# pixelator.pna.anndata or pixelator.pna.sample_calling, but that would create a circular import
# with pixelator.pna.config.panel.diff.

from collections.abc import Mapping
from typing import Any

from anndata import AnnData

from pixelator.pna.config.panel.combination import PNAAntibodyPanelCombination

SAMPLE_CALLING_UNS_KEY = "sample_calling"
SAMPLE_CALLING_COLLAPSED_KEY = "collapsed"
ORIGINAL_HASH_COUNTS_PREFIX = "original_hash_counts_"


def sample_calling_hashing_collapsed(adata: AnnData) -> bool:
    """Return whether hashing markers were collapsed out of ``var``.

    After sample calling, hashing antibodies are stored on ``obs`` and the
    original panel snapshots remain in ``uns``. New files record that layout
    as ``uns["sample_calling"]["collapsed"]``.

    When the key is present it is trusted. When it is missing (legacy
    sample-called files), the layout is inferred from
    ``original_hash_counts_*`` columns on ``obs``, or from hashing clones in
    the stored panels that are absent from ``var``. Otherwise the file is
    treated as not collapsed (``var`` is 1:1 with the stored panels). Missing
    panel snapshots in ``uns`` are treated as not collapsed.
    """
    payload = adata.uns.get(SAMPLE_CALLING_UNS_KEY)
    if isinstance(payload, Mapping) and SAMPLE_CALLING_COLLAPSED_KEY in payload:
        return bool(payload[SAMPLE_CALLING_COLLAPSED_KEY])
    if _has_original_hash_counts(adata):
        return True
    try:
        hashing_ids = PNAAntibodyPanelCombination.from_adata(adata).hashing_marker_ids
    except KeyError:
        return False
    if not hashing_ids:
        return False
    return not hashing_ids.issubset(_var_marker_ids(adata))


def set_sample_calling_collapsed(adata: AnnData, collapsed: bool = True) -> None:
    """Record whether hashing markers are collapsed out of ``adata.var``."""
    payload: dict[str, Any] = {}
    existing = adata.uns.get(SAMPLE_CALLING_UNS_KEY)
    if isinstance(existing, Mapping):
        payload.update(existing)
    payload[SAMPLE_CALLING_COLLAPSED_KEY] = collapsed
    adata.uns[SAMPLE_CALLING_UNS_KEY] = payload


def _has_original_hash_counts(adata: AnnData) -> bool:
    return any(
        str(column).startswith(ORIGINAL_HASH_COUNTS_PREFIX)
        for column in adata.obs.columns
    )


def _var_marker_ids(adata: AnnData) -> set[str]:
    if "marker_id" in adata.var.columns:
        return {str(marker_id) for marker_id in adata.var["marker_id"]}
    return {str(marker_id) for marker_id in adata.var.index}
