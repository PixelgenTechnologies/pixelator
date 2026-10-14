"""Tests for finding markers whose id changed between two panels.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.config.panel.diff import changed_marker_ids


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


def test_changed_marker_ids_returns_renamed_markers():
    old = _panel("0.1.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("0.1.1", [_marker("CD3E", "AAAA"), _marker("CD4", "CCCC")])

    assert changed_marker_ids(old, new) == {"CD3": "CD3E"}


def test_changed_marker_ids_rejects_a_concatenated_panel():
    left = _panel("0.1.0", [_marker("CD3", "AAAA")])
    right = _panel("0.1.0", [_marker("CD19", "CCCC")])
    combined = PNAAntibodyPanel.concatenate([left, right])

    with pytest.raises(ValueError, match="single source"):
        changed_marker_ids(combined, left)
    with pytest.raises(ValueError, match="single source"):
        changed_marker_ids(left, combined)


def test_changed_marker_ids_ignores_markers_present_on_one_side_only():
    old = _panel("0.1.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("0.1.1", [_marker("CD3", "AAAA"), _marker("CD19", "GGGG")])

    assert changed_marker_ids(old, new) == {}
