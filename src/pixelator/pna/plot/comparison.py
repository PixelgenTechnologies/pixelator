"""Plots and HTML reports for comparing abundance/proximity similarity between sample pairs.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pixelator.pna.analysis.comparison import SamplePairComparisonResult


def _plot_similarity_scatter(ax, data, x, y, correlation, xlabel, ylabel, title):
    sns.scatterplot(data=data, x=x, y=y, alpha=0.7, ax=ax)

    lo = min(data[x].min(), data[y].min())
    hi = max(data[x].max(), data[y].max())
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", label="y = x")

    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_sample_pair_comparison(
    result: SamplePairComparisonResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot the abundance and proximity comparison for a sample pair.

    Produces a figure with two scatter plots, both expected to be concentrated
    around the ``y = x`` line if the two samples are similar:

    - Abundance: mean marker CLR value in sample 1 vs. sample 2.
    - Proximity: mean marker-pair proximity log2 ratio in sample 1 vs. sample 2.

    The Pearson correlation is annotated on each subplot.

    Args:
        result: A `SamplePairComparisonResult`, as returned by
            `pixelator.pna.analysis.comparison.compare_sample_pair`.

    Returns:
        A tuple of the created figure and its two
        axes (abundance, proximity).

    """
    sample1_name = result.sample1_name
    sample2_name = result.sample2_name

    fig, (ax_abundance, ax_proximity) = plt.subplots(1, 2, figsize=(14, 6))

    _plot_similarity_scatter(
        ax_abundance,
        result.abundance,
        x=f"mean_clr_{sample1_name}",
        y=f"mean_clr_{sample2_name}",
        correlation=result.abundance_correlation,
        xlabel=f"Mean marker CLR ({sample1_name})",
        ylabel=f"Mean marker CLR ({sample2_name})",
        title="Abundance",
    )
    _plot_similarity_scatter(
        ax_proximity,
        result.proximity,
        x=f"log2_ratio_{sample1_name}",
        y=f"log2_ratio_{sample2_name}",
        correlation=result.proximity_correlation,
        xlabel=f"Mean proximity log2 ratio ({sample1_name})",
        ylabel=f"Mean proximity log2 ratio ({sample2_name})",
        title="Proximity",
    )

    title = f"{sample1_name} vs {sample2_name}"
    if result.gate:
        title += f" (gate: {', '.join(result.gate)})"
    fig.suptitle(title)
    fig.tight_layout()

    return fig, (ax_abundance, ax_proximity)


def _figure_to_base64_png(fig: Figure) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def write_sample_pair_comparison_report(
    results: Sequence[SamplePairComparisonResult],
    output_path: Path | str,
    title: str = "Sample pair abundance/proximity similarity report",
) -> Path:
    """Collect abundance/proximity comparisons for a set of sample pairs into an HTML report.

    For each `SamplePairComparisonResult` in ``results``, this creates the
    abundance/proximity comparison figure (see `plot_sample_pair_comparison`)
    and embeds it, together with the sample names, correlations, and gate (if
    any), into a single self-contained HTML file.

    Args:
        results: The comparison results to include in the report, e.g. as
            returned by `pixelator.pna.analysis.comparison.compare_sample_pairs`
            or `pixelator.pna.analysis.comparison.compare_sample_pairs_by_gate`.
        output_path: The path to write the HTML report to.
        title: The title of the report. Defaults to
            "Sample pair abundance/proximity similarity report".

    Returns:
        The path to the written HTML report.

    """
    output_path = Path(output_path)

    sections = []
    for result in results:
        fig, _ = plot_sample_pair_comparison(result)
        encoded_png = _figure_to_base64_png(fig)
        plt.close(fig)

        sample1_name = escape(result.sample1_name)
        sample2_name = escape(result.sample2_name)

        gate_html = (
            f"<p><strong>Gate:</strong> {escape(', '.join(result.gate))}</p>"
            if result.gate
            else ""
        )
        sections.append(f"""
        <section>
            <h2>{sample1_name} vs {sample2_name}</h2>
            {gate_html}
            <p>
                <strong>Abundance correlation:</strong> {result.abundance_correlation:.3f}
                &nbsp;|&nbsp;
                <strong>Proximity correlation:</strong> {result.proximity_correlation:.3f}
            </p>
            <img src="data:image/png;base64,{encoded_png}"
                 alt="Abundance/proximity comparison for {sample1_name} vs {sample2_name}" />
        </section>
        """)

    escaped_title = escape(title)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>{escaped_title}</title>
<style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    section {{ margin-bottom: 3rem; border-bottom: 1px solid #ccc; padding-bottom: 2rem; }}
    img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
{"".join(sections)}
</body>
</html>
"""

    output_path.write_text(html)
    return output_path
