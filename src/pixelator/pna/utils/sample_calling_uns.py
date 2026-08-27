"""AnnData ``uns`` keys for sample-calling layout.

Kept as a leaf module so panel upgrade code can read the collapsed flag
without importing :mod:`pixelator.pna.anndata` or
:mod:`pixelator.pna.sample_calling` (both load panels).

Copyright © 2025 Pixelgen Technologies AB.
"""

# This is not a great place for these functions they should probably be placed under
# pixelator.pna.anndata or pixelator.pna.sample_calling, but that would create a circular import
# with pixelator.pna.config.panel.diff.

from collections.abc import Mapping
from typing import Any

from anndata import AnnData

SAMPLE_CALLING_UNS_KEY = "sample_calling"
SAMPLE_CALLING_COLLAPSED_KEY = "collapsed"


def sample_calling_hashing_collapsed(adata: AnnData) -> bool:
    """Return whether hashing markers were collapsed out of ``var``.

    After sample calling, hashing antibodies are stored on ``obs`` and the
    original panel snapshots remain in ``uns``. This flag records that
    ``adata.var`` (and the edgelist) are in the collapsed layout.

    Files that never went through sample calling omit the key and are treated
    as not collapsed (``var`` is 1:1 with the stored panels).
    """
    payload = adata.uns.get(SAMPLE_CALLING_UNS_KEY)
    if not isinstance(payload, Mapping):
        return False
    return bool(payload.get(SAMPLE_CALLING_COLLAPSED_KEY, False))


def set_sample_calling_collapsed(adata: AnnData, collapsed: bool = True) -> None:
    """Record whether hashing markers are collapsed out of ``adata.var``."""
    payload: dict[str, Any] = {}
    existing = adata.uns.get(SAMPLE_CALLING_UNS_KEY)
    if isinstance(existing, Mapping):
        payload.update(existing)
    payload[SAMPLE_CALLING_COLLAPSED_KEY] = collapsed
    adata.uns[SAMPLE_CALLING_UNS_KEY] = payload
