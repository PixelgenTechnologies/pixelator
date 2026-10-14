"""Tests for `pixelator.pna.analysis.segmentation.cc_protein_weights`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import inspect

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from pixelator.pna.analysis.segmentation import cc_protein_weights
from pixelator.pna.analysis.segmentation.protein_weights import (
    _plot_protein_weights,
    _scale_columns_to_unit_sum,
)

POP1 = "Mono"
POP2 = "CD4T"
POP1_MARKERS = [f"A{i}" for i in range(1, 7)]
POP2_MARKERS = [f"B{i}" for i in range(1, 7)]
SHARED_MARKER = "HLA-ABC"
NOISE_MARKERS = ["N1", "N2", "N3"]
ALL_MARKERS = POP1_MARKERS + POP2_MARKERS + [SHARED_MARKER] + NOISE_MARKERS


def _two_population_adata(
    n_pop1: int = 20,
    n_pop2: int = 20,
    *,
    seed: int = 0,
) -> ad.AnnData:
    """Build counts with mutually exclusive marker sets plus a shared abundant protein."""
    rng = np.random.default_rng(seed)
    n_obs = n_pop1 + n_pop2
    x = np.zeros((n_obs, len(ALL_MARKERS)), dtype=np.float64)
    marker_index = {name: i for i, name in enumerate(ALL_MARKERS)}

    for i in range(n_pop1):
        for marker in POP1_MARKERS:
            x[i, marker_index[marker]] = rng.integers(8, 15)
        x[i, marker_index[SHARED_MARKER]] = rng.integers(40, 60)
        for marker in NOISE_MARKERS:
            x[i, marker_index[marker]] = rng.integers(1, 3)

    for i in range(n_pop1, n_obs):
        for marker in POP2_MARKERS:
            x[i, marker_index[marker]] = rng.integers(8, 15)
        x[i, marker_index[SHARED_MARKER]] = rng.integers(40, 60)
        for marker in NOISE_MARKERS:
            x[i, marker_index[marker]] = rng.integers(1, 3)

    obs = pd.DataFrame(
        {"cell_type": [POP1] * n_pop1 + [POP2] * n_pop2},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=pd.Index(ALL_MARKERS, name="marker_id"))
    return ad.AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def two_pop_adata() -> ad.AnnData:
    return _two_population_adata()


def test_cc_protein_weights_shape_and_column_names(two_pop_adata):
    weights = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
    )

    assert weights.shape[1] == 2
    assert list(weights.columns) == [POP1, POP2]
    assert weights.shape[0] >= 5
    assert set(weights.index).issubset(set(two_pop_adata.var_names))
    assert (weights >= 0).all().all()
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    assert weights.attrs["model_type"] == "cell_abundance"


def test_cc_protein_weights_assigns_exclusive_markers(two_pop_adata):
    weights = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        random_state=0,
    )

    kept_pop1 = [m for m in POP1_MARKERS if m in weights.index]
    kept_pop2 = [m for m in POP2_MARKERS if m in weights.index]
    assert kept_pop1, "expected population-1 exclusive markers to pass the filter"
    assert kept_pop2, "expected population-2 exclusive markers to pass the filter"
    assert SHARED_MARKER not in weights.index

    assert weights.loc[kept_pop1, POP1].mean() > weights.loc[kept_pop1, POP2].mean()
    assert weights.loc[kept_pop2, POP2].mean() > weights.loc[kept_pop2, POP1].mean()


def test_cc_protein_weights_is_deterministic_with_seed(two_pop_adata):
    first = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        random_state=0,
    )
    second = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        random_state=0,
    )
    pd.testing.assert_frame_equal(first, second)


def test_cc_protein_weights_masked_markers_are_excluded(two_pop_adata):
    masked = POP1_MARKERS[:2]
    weights = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        masked_markers=masked,
    )
    assert not set(masked).intersection(weights.index)


def test_cc_protein_weights_min_diff_one_keeps_exclusive_markers(two_pop_adata):
    """A relative-diff of 1.0 is reachable; min_diff=1.0 must not drop exclusive markers."""
    weights = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        min_diff=1.0,
    )
    kept_exclusive = [m for m in POP1_MARKERS + POP2_MARKERS if m in weights.index]
    assert len(kept_exclusive) >= 5
    assert SHARED_MARKER not in weights.index


def test_cc_protein_weights_too_few_markers_raises(two_pop_adata):
    with pytest.raises(ValueError, match="markers are kept after filtering"):
        cc_protein_weights(
            two_pop_adata,
            group_by="cell_type",
            population_1=POP1,
            population_2=POP2,
            max_freq=0.0,
        )


def test_cc_protein_weights_missing_group_by_raises(two_pop_adata):
    with pytest.raises(ValueError, match="not found in adata.obs"):
        cc_protein_weights(
            two_pop_adata,
            group_by="missing",
            population_1=POP1,
            population_2=POP2,
        )


def test_cc_protein_weights_unknown_population_raises(two_pop_adata):
    with pytest.raises(ValueError, match="not found"):
        cc_protein_weights(
            two_pop_adata,
            group_by="cell_type",
            population_1=POP1,
            population_2="Unknown",
        )


def test_cc_protein_weights_k_neighborhood_not_implemented(two_pop_adata):
    with pytest.raises(NotImplementedError, match="k_neighborhood"):
        cc_protein_weights(
            two_pop_adata,
            group_by="cell_type",
            population_1=POP1,
            population_2=POP2,
            mode="k_neighborhood",
        )


def test_cc_protein_weights_show_plot_default_is_false():
    assert (
        inspect.signature(cc_protein_weights).parameters["show_plot"].default is False
    )


def test_scale_columns_to_unit_sum_keeps_a_zero_column():
    """A zero column stays zero; every other column sums to 1."""
    raw = np.array([[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
    scaled = _scale_columns_to_unit_sum(raw)
    np.testing.assert_allclose(scaled[:, 0], [0.4, 0.4, 0.2])
    np.testing.assert_allclose(scaled[:, 1], 0.0)
    np.testing.assert_array_equal(raw, [[2.0, 0.0], [2.0, 0.0], [1.0, 0.0]])


def test_plot_protein_weights_stacks_by_mean_with_r_labels():
    """Highest mean is at the top, even when another marker has a higher max."""
    weights = pd.DataFrame(
        {"Mono": [0.50, 0.90, 0.10], "CD4T": [0.50, 0.00, 0.20]},
        index=["A", "B", "C"],
    )
    fig, ax = _plot_protein_weights(weights)
    try:
        # Tick labels run bottom to top. Means are A 0.50, B 0.45, C 0.15,
        # so A is at the top. Ordering by max would put B above A.
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["C", "B", "A"]
        assert ax.get_title() == "NMF model weights"
        assert ax.get_xlabel() == "Weights"
        assert ax.get_ylabel() == "Protein"
        legend = ax.get_legend()
        assert legend.get_title().get_text() == "Population"
        assert [text.get_text() for text in legend.get_texts()] == ["Mono", "CD4T"]

        assert len(ax.containers) == 2
        first_widths = [bar.get_width() for bar in ax.containers[0]]
        second_lefts = [bar.get_x() for bar in ax.containers[1]]
        np.testing.assert_allclose(second_lefts, first_widths)
        assert to_hex(ax.containers[0][0].get_facecolor()) == "#6588cf"
        assert to_hex(ax.containers[1][0].get_facecolor()) == "#f19da7"
        fig.canvas.draw()
        assert ax.get_xgridlines()
        assert all(line.get_visible() for line in ax.get_xgridlines())
        assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        ymin, ymax = ax.get_ylim()
        assert -1.0 < ymin < 0
        assert 2 < ymax < 3
    finally:
        plt.close(fig)


def test_plot_protein_weights_keeps_union_of_top_40():
    markers = [f"m{i:02d}" for i in range(90)]
    pop1 = np.zeros(90)
    pop2 = np.zeros(90)
    pop1[:40] = np.linspace(40, 1, 40)
    pop2[40:80] = np.linspace(40, 1, 40)
    pop1[80:] = 0.01
    pop2[80:] = 0.01
    weights = pd.DataFrame({"P1": pop1, "P2": pop2}, index=markers)
    fig, ax = _plot_protein_weights(weights)
    try:
        shown = {tick.get_text() for tick in ax.get_yticklabels()}
        assert shown == set(markers[:80])
        ymin, ymax = ax.get_ylim()
        assert ymin > -1
        assert ymax < 80
    finally:
        plt.close(fig)


def test_cc_protein_weights_show_plot_smoke(two_pop_adata, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    weights = cc_protein_weights(
        two_pop_adata,
        group_by="cell_type",
        population_1=POP1,
        population_2=POP2,
        show_plot=True,
    )
    assert list(weights.columns) == [POP1, POP2]
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "NMF model weights"
    assert len(ax.containers) == 2
    plt.close("all")


def test_cc_protein_weights_exported_from_analysis():
    from pixelator.pna.analysis import cc_protein_weights as exported

    assert exported is cc_protein_weights
