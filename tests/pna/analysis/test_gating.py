"""Tests for the pna gating module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import pytest

from pixelator.pna.analysis.gating import (
    determine_marker_threshold,
    gate_mask,
    parse_gate,
)


@pytest.mark.parametrize(
    "gate,expected",
    [
        (["+CD3e"], [("CD3e", "+")]),
        (["-CD19"], [("CD19", "-")]),
        (["+CD3e", "+CD4", "-CD19"], [("CD3e", "+"), ("CD4", "+"), ("CD19", "-")]),
    ],
)
def test_parse_gate(gate, expected):
    """Verify gate strings are parsed into (marker, sign) tuples."""
    assert parse_gate(gate) == expected


@pytest.mark.parametrize("invalid_gate", [["CD3e"], ["*CD3e"], [""]])
def test_parse_gate_invalid(invalid_gate):
    """Verify an invalid gate specification raises a ValueError."""
    with pytest.raises(ValueError):
        parse_gate(invalid_gate)


def test_determine_marker_threshold_bimodal():
    """Verify a clearly bimodal distribution yields a threshold between the two modes."""
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(0, 0.2, 200), rng.normal(6, 0.2, 200)],
    )

    result = determine_marker_threshold(values, min_separation_score=3.0, marker="A")

    assert result.marker == "A"
    assert result.threshold is not None
    assert 0 < result.threshold < 6
    assert result.separation_score >= 3.0


def test_determine_marker_threshold_unimodal_is_ignored(caplog):
    """Verify a unimodal distribution yields no threshold and logs a warning."""
    rng = np.random.default_rng(0)
    values = rng.normal(2, 0.5, 400)

    result = determine_marker_threshold(values, min_separation_score=3.0, marker="B")

    assert result.threshold is None
    assert result.separation_score < 3.0
    assert any("unimodal" in message.lower() for message in caplog.messages)


def test_gate_mask_positive_and_negative():
    """Verify gate_mask combines positive and negative marker criteria."""
    rng = np.random.default_rng(0)
    n = 200
    clr = pd.DataFrame(
        {
            "A": np.concatenate(
                [rng.normal(0, 0.2, n // 2), rng.normal(6, 0.2, n // 2)]
            ),
            "B": np.concatenate(
                [rng.normal(0, 0.2, n // 2), rng.normal(6, 0.2, n // 2)]
            ),
        },
        index=[f"c{i}" for i in range(n)],
    )

    mask, thresholds = gate_mask(clr, ["+A", "-B"], min_separation_score=3.0)

    assert set(mask.index) == set(clr.index)
    assert (
        mask
        == (
            (clr["A"] >= thresholds[0].threshold) & (clr["B"] < thresholds[1].threshold)
        )
    ).all()
    assert mask.sum() == 0  # no component is both high-A and low-B in this construction


def test_gate_mask_ignores_unimodal_marker():
    """Verify a unimodal marker in the gate is ignored (does not constrain the mask)."""
    rng = np.random.default_rng(0)
    n = 200
    clr = pd.DataFrame(
        {
            "A": np.concatenate(
                [rng.normal(0, 0.2, n // 2), rng.normal(6, 0.2, n // 2)]
            ),
            "B": rng.normal(2, 0.5, n),
        },
        index=[f"c{i}" for i in range(n)],
    )

    mask, thresholds = gate_mask(clr, ["+A", "+B"], min_separation_score=3.0)

    b_threshold = next(t for t in thresholds if t.marker == "B")
    assert b_threshold.threshold is None

    a_threshold = next(t for t in thresholds if t.marker == "A")
    expected_mask = clr["A"] >= a_threshold.threshold
    assert (mask == expected_mask).all()


def test_gate_mask_missing_marker_raises():
    """Verify gating on a marker missing from the CLR data raises a KeyError."""
    clr = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["c1", "c2", "c3"])

    with pytest.raises(KeyError):
        gate_mask(clr, ["+NotAMarker"])
