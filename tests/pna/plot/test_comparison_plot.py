"""Tests for the pna sample pair comparison plot/report module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pixelator.pna.analysis.comparison import SamplePairComparisonResult
from pixelator.pna.plot import (
    plot_sample_pair_comparison,
    write_sample_pair_comparison_report,
)


@pytest.fixture(name="comparison_result")
def comparison_result_fixture():
    """A synthetic `SamplePairComparisonResult`, independent of any real pxl data."""
    abundance = pd.DataFrame(
        {
            "marker": ["MarkerA", "MarkerB", "MarkerC"],
            "mean_clr_sample1": [1.0, 2.0, 3.0],
            "mean_clr_sample2": [1.1, 2.1, 2.9],
        }
    )
    proximity = pd.DataFrame(
        {
            "marker_1": ["MarkerA", "MarkerB"],
            "marker_2": ["MarkerA", "MarkerC"],
            "log2_ratio_sample1": [0.5, -0.2],
            "n_cells_sample1": [80, 90],
            "log2_ratio_sample2": [0.4, -0.3],
            "n_cells_sample2": [70, 95],
        }
    )
    return SamplePairComparisonResult(
        sample1_name="sample1",
        sample2_name="sample2",
        abundance=abundance,
        abundance_correlation=0.98,
        proximity=proximity,
        proximity_correlation=0.91,
    )


def test_plot_sample_pair_comparison(comparison_result):
    """Verify the comparison plot has two labeled scatter subplots."""
    fig, (ax_abundance, ax_proximity) = plot_sample_pair_comparison(comparison_result)

    assert isinstance(fig, plt.Figure)
    assert ax_abundance.get_title() == "Abundance"
    assert ax_proximity.get_title() == "Proximity"
    assert "sample1" in ax_abundance.get_xlabel()
    assert "sample2" in ax_abundance.get_ylabel()

    plt.close(fig)


def test_plot_sample_pair_comparison_with_gate(comparison_result):
    """Verify the gate is reflected in the figure title when present."""
    comparison_result.gate = ["+MarkerA", "-MarkerB"]
    fig, _ = plot_sample_pair_comparison(comparison_result)

    assert "+MarkerA" in fig.get_suptitle()
    plt.close(fig)


def test_write_sample_pair_comparison_report(comparison_result, tmp_path):
    """Verify the HTML report embeds sample names, correlations, and an image per result."""
    output_path = tmp_path / "report.html"

    result_path = write_sample_pair_comparison_report(
        [comparison_result], output_path, title="My report"
    )

    assert result_path == output_path
    html = output_path.read_text()

    assert "My report" in html
    assert "sample1 vs sample2" in html
    assert "0.980" in html
    assert "0.910" in html
    assert html.count("<img") == 1
    assert "data:image/png;base64," in html


def test_write_sample_pair_comparison_report_multiple_results_and_gate(
    comparison_result, tmp_path
):
    """Verify multiple results are all included, and a gate is rendered when present."""
    gated_result = SamplePairComparisonResult(
        sample1_name="sample3",
        sample2_name="sample4",
        abundance=comparison_result.abundance.rename(
            columns={
                "mean_clr_sample1": "mean_clr_sample3",
                "mean_clr_sample2": "mean_clr_sample4",
            }
        ),
        abundance_correlation=0.5,
        proximity=comparison_result.proximity.rename(
            columns={
                "log2_ratio_sample1": "log2_ratio_sample3",
                "n_cells_sample1": "n_cells_sample3",
                "log2_ratio_sample2": "log2_ratio_sample4",
                "n_cells_sample2": "n_cells_sample4",
            }
        ),
        proximity_correlation=0.4,
        gate=["+MarkerA", "-MarkerB"],
    )

    output_path = tmp_path / "report.html"
    write_sample_pair_comparison_report([comparison_result, gated_result], output_path)

    html = output_path.read_text()
    assert html.count("<img") == 2
    assert "sample1 vs sample2" in html
    assert "sample3 vs sample4" in html
    assert "+MarkerA, -MarkerB" in html
