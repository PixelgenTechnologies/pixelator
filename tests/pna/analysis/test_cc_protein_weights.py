"""Tests for `pixelator.pna.analysis.segmentation.cc_protein_weights`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import inspect

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pixelator.pna.analysis.segmentation import cc_protein_weights

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


def test_cc_protein_weights_too_few_markers_raises(two_pop_adata):
    with pytest.raises(ValueError, match="markers are kept after filtering"):
        cc_protein_weights(
            two_pop_adata,
            group_by="cell_type",
            population_1=POP1,
            population_2=POP2,
            min_diff=1.0,
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


def test_cc_protein_weights_exported_from_analysis():
    from pixelator.pna.analysis import cc_protein_weights as exported

    assert exported is cc_protein_weights
