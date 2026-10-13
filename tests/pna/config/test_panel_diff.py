"""Tests for changed marker ids and AnnData panel upgrades.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.config.panel.diff import changed_marker_ids, upgrade_adata

_STORED_COLUMNS = ["control", "sequence_1", "sequence_2", "uniprot_id"]


def _marker(marker_id: str, sequence: str, **extra) -> dict:
    return {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
        **extra,
    }


def _panel(version: str, rows: list[dict]) -> PNAAntibodyPanel:
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(name="panel", version=version, product="kit"),
    )


def _adata(panel: PNAAntibodyPanel) -> AnnData:
    """Build an AnnData whose stored panel columns lack the optional ones."""
    var = panel.df[_STORED_COLUMNS].copy()
    var["positive_cells_count"] = range(len(var))
    adata = AnnData(
        X=np.arange(2 * len(var), dtype=float).reshape(2, len(var)),
        obs=pd.DataFrame(index=["a", "b"]),
        var=var,
    )
    adata.uns["panel_metadata"] = {
        **panel.metadata.model_dump(),
        "panel_columns": list(_STORED_COLUMNS),
    }
    return adata


def test_changed_marker_ids_returns_renamed_markers():
    old = _panel("0.1.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("0.1.1", [_marker("CD3E", "AAAA"), _marker("CD4", "CCCC")])

    assert changed_marker_ids(old, new) == {"CD3": "CD3E"}


def test_changed_marker_ids_ignores_markers_present_on_one_side_only():
    old = _panel("0.1.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("0.1.1", [_marker("CD3", "AAAA"), _marker("CD19", "GGGG")])

    assert changed_marker_ids(old, new) == {}


def test_upgrade_adata_keeps_metadata_and_var_consistent():
    old = _panel(
        "0.1.0",
        [_marker("CD3", "AAAA", uniprot_id=""), _marker("CD4", "CCCC", uniprot_id="")],
    )
    new = _panel(
        "0.1.1",
        [
            _marker("CD4", "CCCC", uniprot_id="P01730"),
            _marker("CD3E", "AAAA", uniprot_id="P07766"),
        ],
    )
    adata = _adata(old)
    counts = adata.X.copy()

    upgraded = upgrade_adata(adata, new)

    assert list(upgraded.var.index) == ["CD3E", "CD4"]
    assert upgraded.var["uniprot_id"].tolist() == ["P07766", "P01730"]
    assert upgraded.var["positive_cells_count"].tolist() == [0, 1]
    assert (upgraded.X == counts).all()
    assert upgraded.uns["panel_metadata"]["version"] == "0.1.1"
    assert PNAAntibodyPanel.from_adata(upgraded) == new


def test_upgrade_adata_rejects_panels_with_different_markers():
    old = _panel("0.1.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("0.1.1", [_marker("CD3", "AAAA"), _marker("CD19", "GGGG")])

    with pytest.raises(ValueError, match="same markers"):
        upgrade_adata(_adata(old), new)
