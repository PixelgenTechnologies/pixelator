"""Tests for sample-calling collapsed-layout detection.

Copyright © 2025 Pixelgen Technologies AB.
"""

import pandas as pd
from anndata import AnnData

from pixelator.pna.anndata import add_panel_information
from pixelator.pna.config.panel import PNAAntibodyPanelCombination
from pixelator.pna.utils.sample_calling_uns import (
    ORIGINAL_HASH_COUNTS_PREFIX,
    sample_calling_hashing_collapsed,
    set_sample_calling_collapsed,
)


def test_sample_calling_hashing_collapsed_trusts_explicit_flag(panel, hashing_panel):
    """An explicit collapsed key wins over layout heuristics."""
    combo = PNAAntibodyPanelCombination([panel.partial_panels()[0], hashing_panel])
    adata = add_panel_information(
        AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=pd.Index(list(combo.markers), name="marker_id")),
        ),
        combo,
    )
    assert sample_calling_hashing_collapsed(adata) is False

    set_sample_calling_collapsed(adata, True)
    assert sample_calling_hashing_collapsed(adata) is True

    sliced = adata[:, list(panel.partial_panels()[0].markers)].copy()
    set_sample_calling_collapsed(sliced, False)
    assert sample_calling_hashing_collapsed(sliced) is False


def test_sample_calling_hashing_collapsed_infers_from_missing_hashing_clones(
    panel, hashing_panel
):
    """Hashing clones stored in uns but missing from var imply collapsed layout."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    adata = add_panel_information(
        AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=pd.Index(list(combo.markers), name="marker_id")),
        ),
        combo,
    )
    adata = adata[:, list(base.markers)].copy()
    assert "sample_calling" not in adata.uns
    assert sample_calling_hashing_collapsed(adata) is True


def test_sample_calling_hashing_collapsed_infers_from_original_hash_counts(
    panel, hashing_panel
):
    """original_hash_counts_* is a legacy proxy for sample calling."""
    combo = PNAAntibodyPanelCombination([panel.partial_panels()[0], hashing_panel])
    adata = add_panel_information(
        AnnData(
            obs=pd.DataFrame(
                {f"{ORIGINAL_HASH_COUNTS_PREFIX}HM-1": [1.0]},
                index=["c1"],
            ),
            var=pd.DataFrame(index=pd.Index(list(combo.markers), name="marker_id")),
        ),
        combo,
    )
    assert sample_calling_hashing_collapsed(adata) is True


def test_sample_calling_hashing_collapsed_false_without_hashing_panel(panel):
    """A 1:1 var with no hashing panel is not collapsed."""
    adata = add_panel_information(
        AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=pd.Index(list(panel.markers), name="marker_id")),
        ),
        panel,
    )
    assert sample_calling_hashing_collapsed(adata) is False


def test_sample_calling_hashing_collapsed_false_without_panel_uns():
    """Missing panel snapshots must not raise; treat as not collapsed."""
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["M1"], name="marker_id")),
    )
    assert sample_calling_hashing_collapsed(adata) is False
