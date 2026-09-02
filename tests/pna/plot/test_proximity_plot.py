"""Tests for the pna proximity heatmap plot module.

Copyright © 2026 Pixelgen Technologies AB.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from pixelator.pna.plot import proximity_heatmap
from pixelator.pna.plot.proximity import _neg_log10_p


@pytest.fixture(name="proximity_data")
def proximity_data_fixture():
    """A small, one-direction-per-pair marker-pair summary table."""
    return pd.DataFrame(
        {
            "marker_1": ["CD3", "CD3", "CD3", "CD4", "CD4", "CD8"],
            "marker_2": ["CD3", "CD4", "CD8", "CD4", "CD8", "CD8"],
            "mean_log2_ratio": [0.0, 0.8, -0.5, 0.0, 0.3, 0.0],
            "p_adjusted": [1.0, 0.001, 0.2, 1.0, 0.04, 1.0],
        }
    )


def test_tiles_returns_figure_and_axes(proximity_data):
    """The default 'tiles' plot returns a Figure and the heatmap Axes."""
    fig, ax = proximity_heatmap(proximity_data)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_dots_returns_figure_and_axes(proximity_data):
    """The 'dots' plot returns a Figure and Axes too."""
    fig, ax = proximity_heatmap(proximity_data, kind="dots")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="./snapshots/test_tiles_heatmap_image",
)
def test_tiles_heatmap_image(proximity_data):
    """Regression test guarding the rendered appearance of the 'tiles' heatmap."""
    fig, _ = proximity_heatmap(
        proximity_data,
        highlight_pairs=pd.DataFrame({"marker_1": ["CD3"], "marker_2": ["CD8"]}),
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="./snapshots/test_dots_heatmap_image",
)
def test_dots_heatmap_image(proximity_data):
    """Regression test guarding the rendered appearance of the 'dots' heatmap."""
    fig, _ = proximity_heatmap(
        proximity_data,
        kind="dots",
        highlight_pairs=pd.DataFrame({"marker_1": ["CD3"], "marker_2": ["CD8"]}),
    )
    return fig


def test_tiles_return_plot_data_is_symmetrised(proximity_data):
    """With symmetrise=True (default), the pivoted matrix is filled on both sides."""
    wide = proximity_heatmap(proximity_data, return_plot_data=True)
    assert wide.loc["CD3", "CD8"] == wide.loc["CD8", "CD3"] == -0.5
    assert set(wide.index) == set(wide.columns) == {"CD3", "CD4", "CD8"}


def test_symmetrise_false_leaves_missing_direction_as_nan(proximity_data):
    """With symmetrise=False, only the given direction of each pair is filled in."""
    wide = proximity_heatmap(proximity_data, symmetrise=False, return_plot_data=True)
    assert wide.loc["CD3", "CD8"] == -0.5
    assert pd.isna(wide.loc["CD8", "CD3"])


def test_dots_return_plot_data_has_ordered_categoricals(proximity_data):
    """return_plot_data for 'dots' gives marker columns as ordered categoricals."""
    long_data = proximity_heatmap(proximity_data, kind="dots", return_plot_data=True)
    assert isinstance(long_data["marker_1"].dtype, pd.CategoricalDtype)
    assert long_data["marker_1"].cat.ordered
    assert set(long_data["marker_1"].cat.categories) == {"CD3", "CD4", "CD8"}


def test_no_clustering_keeps_alphabetical_order(proximity_data):
    """Without clustering, markers appear in sorted order along both axes."""
    long_data = proximity_heatmap(
        proximity_data,
        kind="dots",
        cluster_rows=False,
        cluster_cols=False,
        return_plot_data=True,
    )
    assert list(long_data["marker_1"].cat.categories) == ["CD3", "CD4", "CD8"]
    assert list(long_data["marker_2"].cat.categories) == ["CD3", "CD4", "CD8"]


def test_size_col_transform_renames_column(proximity_data):
    """size_col_transform is applied and produces a '{size_col}_transformed' column."""
    long_data = proximity_heatmap(
        proximity_data,
        kind="dots",
        size_col_transform=lambda p: -np.log10(p),
        return_plot_data=True,
    )
    assert "p_adjusted_transformed" in long_data.columns
    np.testing.assert_allclose(
        long_data.loc[long_data["p_adjusted"] == 0.001, "p_adjusted_transformed"], 3.0
    )


def test_default_size_col_transform_is_neg_log10(proximity_data):
    """Default dots size mapping uses -log10 so smaller p-values become larger dots."""
    long_data = proximity_heatmap(proximity_data, kind="dots", return_plot_data=True)
    assert "p_adjusted_transformed" in long_data.columns
    np.testing.assert_allclose(
        long_data["p_adjusted_transformed"], _neg_log10_p(long_data["p_adjusted"])
    )

    data = proximity_data.copy()
    fig, ax = proximity_heatmap(
        data, kind="dots", symmetrise=False, cluster_rows=False, cluster_cols=False
    )
    sizes_by_pair = {
        (m1, m2): s
        for m1, m2, s in zip(
            data["marker_1"],
            data["marker_2"],
            ax.collections[0].get_sizes(),
            strict=True,
        )
    }
    # Most significant pair (p=0.001) must be larger than least significant (p=1.0).
    assert sizes_by_pair[("CD3", "CD4")] > sizes_by_pair[("CD3", "CD3")]
    plt.close(fig)


def test_size_col_transform_none_maps_raw_values_linearly(proximity_data):
    """Passing size_col_transform=None keeps a linear map of the raw column."""
    long_data = proximity_heatmap(
        proximity_data,
        kind="dots",
        size_col_transform=None,
        return_plot_data=True,
    )
    assert "p_adjusted_transformed" not in long_data.columns
    assert "p_adjusted" in long_data.columns


def test_size_col_transform_keeps_original_legend_title(proximity_data):
    """The size legend title stays ``size_col``, not ``{size_col}_transformed``."""
    fig, _ = proximity_heatmap(
        proximity_data,
        kind="dots",
        size_col_transform=lambda p: -np.log10(p),
    )
    assert fig.legends
    assert fig.legends[0].get_title().get_text() == "p_adjusted"
    plt.close(fig)


def test_tiles_ignores_size_col_transform(proximity_data):
    """kind='tiles' does not look up size_col, even when a transform is given."""
    tiles_data = proximity_data.drop(columns=["p_adjusted"])
    wide = proximity_heatmap(
        tiles_data,
        kind="tiles",
        size_col_transform=lambda p: -np.log10(p),
        return_plot_data=True,
    )
    assert set(wide.index) == {"CD3", "CD4", "CD8"}


def test_one_marker_tiles_does_not_raise():
    """A 1x1 tiles heatmap skips clustering instead of calling linkage."""
    data = pd.DataFrame(
        {"marker_1": ["CD3"], "marker_2": ["CD3"], "mean_log2_ratio": [0.1]}
    )
    fig, ax = proximity_heatmap(data, kind="tiles")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_one_marker_dots_does_not_raise():
    """A 1-marker dots plot skips clustering the same way tiles does."""
    data = pd.DataFrame(
        {
            "marker_1": ["CD3"],
            "marker_2": ["CD3"],
            "mean_log2_ratio": [0.1],
            "p_adjusted": [0.05],
        }
    )
    fig, ax = proximity_heatmap(data, kind="dots")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_custom_legend_range_sets_color_normalization(proximity_data):
    """A custom legend_range is used as the vmin/vmax of the dot color scale."""
    fig, ax = proximity_heatmap(proximity_data, kind="dots", legend_range=(-2.0, 2.0))
    scatter = ax.collections[0]
    assert scatter.norm.vmin == -2.0
    assert scatter.norm.vmax == 2.0
    plt.close(fig)


def test_constant_size_col_uses_midpoint_and_single_legend_entry(proximity_data):
    """Identical size_col values map to midpoint size with one legend entry."""
    data = proximity_data.copy()
    data["p_adjusted"] = 0.05
    size_range = (20.0, 300.0)
    fig, ax = proximity_heatmap(
        data,
        kind="dots",
        size_col="p_adjusted",
        size_col_transform=None,
        size_range=size_range,
    )
    scatter = ax.collections[0]
    expected_size = float(np.mean(size_range))
    np.testing.assert_allclose(scatter.get_sizes(), expected_size)

    size_legend = next(
        legend
        for legend in fig.legends
        if legend.get_title().get_text() == "p_adjusted"
    )
    labels = [text.get_text() for text in size_legend.get_texts()]
    assert labels == ["0.05"]
    handle_sizes = [
        handle.get_markersize() ** 2 for handle in size_legend.legend_handles
    ]
    np.testing.assert_allclose(handle_sizes, expected_size)
    plt.close(fig)


def test_dots_figsize_scales_layout_to_fill_figure(proximity_data):
    """A custom figsize scales axes/colorbar placement instead of clipping."""
    fig_auto, _ = proximity_heatmap(proximity_data, kind="dots")
    auto_size = fig_auto.get_size_inches()
    plt.close(fig_auto)

    # Deliberately undersized relative to the auto layout; without scaling,
    # colorbar/axes would be placed past the figure edge in absolute inches.
    target = (max(auto_size[0] * 0.5, 1.0), max(auto_size[1] * 0.5, 1.0))
    fig, ax = proximity_heatmap(proximity_data, kind="dots", figsize=target)
    np.testing.assert_allclose(fig.get_size_inches(), target, atol=1e-2)
    assert len(fig.axes) >= 2
    for artist_ax in fig.axes:
        pos = artist_ax.get_position()
        assert pos.x0 >= -1e-6
        assert pos.y0 >= -1e-6
        assert pos.x1 <= 1.0 + 1e-6
        assert pos.y1 <= 1.0 + 1e-6
    # Plot area should occupy a meaningful fraction of the figure, not a
    # fixed absolute-inch pocket in the corner.
    assert ax.get_position().width > 0.2
    assert ax.get_position().height > 0.2
    plt.close(fig)


def test_highlight_pairs_draws_rectangles(proximity_data):
    """highlight_pairs adds rectangle outlines to the plot."""
    highlight = pd.DataFrame({"marker_1": ["CD3"], "marker_2": ["CD4"]})

    fig_no_highlight, ax_no_highlight = proximity_heatmap(proximity_data, kind="dots")
    fig_highlight, ax_highlight = proximity_heatmap(
        proximity_data, kind="dots", highlight_pairs=highlight
    )

    assert not any(isinstance(p, Rectangle) for p in ax_no_highlight.patches)
    assert any(isinstance(p, Rectangle) for p in ax_highlight.patches)

    plt.close(fig_no_highlight)
    plt.close(fig_highlight)


def test_highlight_color_col_maps_colors(proximity_data):
    """highlight_color_col looks up a per-pair edge color from highlight_colors."""
    highlight = pd.DataFrame(
        {"marker_1": ["CD3", "CD4"], "marker_2": ["CD4", "CD8"], "grp": ["a", "b"]}
    )
    fig, ax = proximity_heatmap(
        proximity_data,
        kind="dots",
        highlight_pairs=highlight,
        highlight_color_col="grp",
        highlight_colors={"a": "red", "b": "blue"},
    )
    edge_colors = {p.get_edgecolor() for p in ax.patches if isinstance(p, Rectangle)}
    assert len(edge_colors) == 2
    plt.close(fig)


def test_highlight_color_col_missing_color_raises(proximity_data):
    """A highlight_color_col value with no entry in highlight_colors raises ValueError."""
    highlight = pd.DataFrame(
        {"marker_1": ["CD3"], "marker_2": ["CD4"], "grp": ["unmapped"]}
    )
    with pytest.raises(ValueError, match="no color for value"):
        proximity_heatmap(
            proximity_data,
            highlight_pairs=highlight,
            highlight_color_col="grp",
            highlight_colors={"a": "red"},
        )


def test_missing_required_column_raises(proximity_data):
    """A DataFrame missing a required column raises ValueError."""
    with pytest.raises(ValueError, match="missing required column"):
        proximity_heatmap(proximity_data.drop(columns=["mean_log2_ratio"]))


def test_duplicate_ordered_pair_raises(proximity_data):
    """Duplicate rows for the same ordered marker pair raise ValueError."""
    duplicated = pd.concat(
        [proximity_data, proximity_data.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicate rows"):
        proximity_heatmap(duplicated)


def test_ambiguous_both_directions_with_symmetrise_raises():
    """Providing both (A, B) and (B, A) with symmetrise=True is ambiguous."""
    data = pd.DataFrame(
        {
            "marker_1": ["A", "B"],
            "marker_2": ["B", "A"],
            "mean_log2_ratio": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="ambiguous"):
        proximity_heatmap(data)


def test_ambiguous_both_directions_ok_with_symmetrise_false():
    """The same input is fine once symmetrise=False."""
    data = pd.DataFrame(
        {
            "marker_1": ["A", "B"],
            "marker_2": ["B", "A"],
            "mean_log2_ratio": [1.0, 2.0],
        }
    )
    fig, ax = proximity_heatmap(data, symmetrise=False)
    plt.close(fig)


def test_invalid_kind_raises(proximity_data):
    """An unknown 'kind' value raises ValueError."""
    with pytest.raises(ValueError, match="'kind'"):
        proximity_heatmap(proximity_data, kind="bars")


def test_invalid_highlight_shrink_raises(proximity_data):
    """highlight_shrink outside [0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="highlight_shrink"):
        proximity_heatmap(proximity_data, highlight_shrink=1.0)
