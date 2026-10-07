"""Tests for table-level sample abundance and proximity comparison.

Copyright © 2026 Pixelgen Technologies AB.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from pixelator.pna.analysis.sample_comparison import (
    AbundanceComparison,
    ProximityComparison,
    compare_abundance,
    compare_proximity,
)


def test_compare_abundance_correlates_mean_clr():
    """Verify mean CLR columns and Pearson correlation for two CLR frames."""
    clr_a = pd.DataFrame({"CD3": [1.0, 3.0], "CD4": [0.0, 2.0], "CD19": [4.0, 4.0]})
    clr_b = pd.DataFrame(
        {"CD3": [2.0, 6.0], "CD4": [1.0, 1.0], "CD8": [9.0, 9.0], "CD19": [0.0, 2.0]}
    )

    result = compare_abundance(clr_a, clr_b, name_a="sample1", name_b="sample2")

    assert isinstance(result, AbundanceComparison)
    assert list(result.abundance["marker"]) == ["CD19", "CD3", "CD4"]
    assert result.abundance["mean_clr_sample1"].tolist() == pytest.approx(
        [4.0, 2.0, 1.0]
    )
    assert result.abundance["mean_clr_sample2"].tolist() == pytest.approx(
        [1.0, 4.0, 1.0]
    )
    expected = result.abundance["mean_clr_sample1"].corr(
        result.abundance["mean_clr_sample2"]
    )
    assert result.correlation == pytest.approx(expected)


def test_compare_abundance_restricts_to_requested_markers():
    """Verify an explicit marker set limits the abundance table."""
    clr_a = pd.DataFrame({"CD3": [1.0, 3.0], "CD4": [0.0, 4.0], "CD19": [2.0, 6.0]})
    clr_b = pd.DataFrame({"CD3": [0.0, 2.0], "CD4": [1.0, 5.0], "CD19": [2.0, 8.0]})

    result = compare_abundance(
        clr_a, clr_b, name_a="a", name_b="b", markers={"CD3", "CD19"}
    )

    assert list(result.abundance["marker"]) == ["CD19", "CD3"]


def test_compare_abundance_requires_distinct_names():
    """Verify identical sample labels are rejected before a column collision."""
    clr = pd.DataFrame({"CD3": [1.0, 2.0], "CD4": [0.0, 3.0]})
    with pytest.raises(ValueError, match="must be different"):
        compare_abundance(clr, clr, name_a="same", name_b="same")


def test_compare_abundance_undefined_correlation():
    """Verify a single marker cannot produce an abundance correlation."""
    clr_a = pd.DataFrame({"CD3": [1.0, 2.0]})
    clr_b = pd.DataFrame({"CD3": [3.0, 4.0]})
    with pytest.raises(ValueError, match="Abundance correlation is undefined"):
        compare_abundance(clr_a, clr_b)


def test_compare_proximity_filters_and_correlates():
    """Verify join-count and cell-count filters, then an inner-join correlation."""
    prox_a = pd.DataFrame(
        {
            "marker_1": ["A", "A", "C", "C", "E", "G", "G"],
            "marker_2": ["B", "B", "D", "D", "F", "H", "H"],
            "log2_ratio": [1.0, 3.0, 4.0, 6.0, 9.0, 1.0, 1.0],
            "join_count_expected_mean": [10, 10, 10, 10, 10, 1, 1],
        }
    )
    prox_b = pd.DataFrame(
        {
            "marker_1": ["A", "A", "C", "C", "E", "E"],
            "marker_2": ["B", "B", "D", "D", "F", "F"],
            "log2_ratio": [2.0, 4.0, 5.0, 7.0, 0.0, 0.0],
            "join_count_expected_mean": [10, 10, 10, 10, 10, 10],
        }
    )

    result = compare_proximity(
        prox_a,
        prox_b,
        name_a="sample1",
        name_b="sample2",
        min_expected_join_count=10,
        min_n_cells=2,
    )

    assert isinstance(result, ProximityComparison)
    pairs = list(zip(result.proximity["marker_1"], result.proximity["marker_2"]))
    assert pairs == [("A", "B"), ("C", "D")]
    assert result.proximity["log2_ratio_sample1"].tolist() == pytest.approx([2.0, 5.0])
    assert result.proximity["log2_ratio_sample2"].tolist() == pytest.approx([3.0, 6.0])
    assert result.proximity["n_cells_sample1"].tolist() == [2, 2]
    assert result.correlation == pytest.approx(1.0)


def test_compare_proximity_no_pairs_pass():
    """Verify an empty inner join raises the proximity filter error."""
    prox_a = pd.DataFrame(
        {
            "marker_1": ["A"],
            "marker_2": ["B"],
            "log2_ratio": [1.0],
            "join_count_expected_mean": [0],
        }
    )
    prox_b = prox_a.copy()
    with pytest.raises(ValueError, match="No marker pairs passed"):
        compare_proximity(prox_a, prox_b, min_expected_join_count=10, min_n_cells=1)


def test_compare_proximity_undefined_correlation():
    """Verify a single shared marker pair cannot produce a correlation."""
    prox = pd.DataFrame(
        {
            "marker_1": ["A", "A"],
            "marker_2": ["B", "B"],
            "log2_ratio": [1.0, 3.0],
            "join_count_expected_mean": [10, 10],
        }
    )
    with pytest.raises(ValueError, match="Proximity correlation is undefined"):
        compare_proximity(prox, prox, min_expected_join_count=0, min_n_cells=1)


def test_sample_comparison_module_does_not_import_pixeldataset():
    """Verify the frame-function module has no pixeldataset import."""
    import pixelator.pna.analysis.sample_comparison as sample_comparison

    tree = ast.parse(Path(sample_comparison.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert set(imported) <= {"__future__", "dataclasses", "pandas"}
    assert not any("pixeldataset" in name for name in imported)
