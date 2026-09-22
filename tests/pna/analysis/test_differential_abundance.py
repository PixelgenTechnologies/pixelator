"""Smoke tests for `pixelator.pna.analysis.differential_abundance`.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pixelator.pna.analysis import differential_abundance

EXPECTED_COLUMNS = [
    "marker",
    "p",
    "p_adj",
    "difference",
    "pct_1",
    "pct_2",
    "target",
    "reference",
]


def _tiny_adata(
    *, with_negatives: bool = False, with_cell_type: bool = False
) -> AnnData:
    rng = np.random.default_rng(0)
    n_per_group = 12
    n_cells = n_per_group * 2
    n_markers = 4
    x = rng.poisson(5, size=(n_cells, n_markers)).astype(float)
    # Plant a higher abundance for M0 in the treated group.
    x[:n_per_group, 0] += 8
    if with_negatives:
        x -= x.mean(axis=1, keepdims=True)

    obs = pd.DataFrame(
        {
            "condition": ["treated"] * n_per_group + ["control"] * n_per_group,
        },
        index=[f"c{i}" for i in range(n_cells)],
    )
    if with_cell_type:
        obs["cell_type"] = (["T"] * (n_per_group // 2) + ["B"] * (n_per_group // 2)) * 2
    var = pd.DataFrame(index=[f"M{i}" for i in range(n_markers)])
    return AnnData(X=x, obs=obs, var=var)


def test_differential_abundance_returns_expected_columns():
    adata = _tiny_adata()
    result = differential_abundance(
        adata,
        contrast_column="condition",
        reference="control",
        targets="treated",
    )

    assert list(result.columns) == EXPECTED_COLUMNS
    assert set(result["marker"]) == {"M0", "M1", "M2", "M3"}
    assert (result["target"] == "treated").all()
    assert (result["reference"] == "control").all()
    planted = result.set_index("marker").loc["M0"]
    assert planted["difference"] > 0


def test_differential_abundance_layer_from_obsm():
    adata = _tiny_adata()
    adata.obsm["clr"] = pd.DataFrame(
        adata.X, index=adata.obs_names, columns=adata.var_names
    )
    result = differential_abundance(
        adata,
        contrast_column="condition",
        reference="control",
        targets="treated",
        layer="clr",
    )

    assert list(result.columns) == EXPECTED_COLUMNS
    assert result.set_index("marker").loc["M0", "difference"] > 0


def test_differential_abundance_group_vars_splits_rows():
    adata = _tiny_adata(with_cell_type=True)
    result = differential_abundance(
        adata,
        contrast_column="condition",
        reference="control",
        targets=["treated"],
        group_vars="cell_type",
    )

    assert list(result.columns) == [*EXPECTED_COLUMNS, "cell_type"]
    assert set(result["cell_type"]) == {"T", "B"}
    assert set(result["marker"]) == {"M0", "M1", "M2", "M3"}
    assert len(result) == 8
    pair_counts = result.groupby(["marker", "cell_type"]).size()
    assert (pair_counts == 1).all()


def test_differential_abundance_invalid_reference_raises():
    adata = _tiny_adata()
    with pytest.raises(
        ValueError, match="reference 'resting' is not present in adata.obs"
    ):
        differential_abundance(
            adata,
            contrast_column="condition",
            reference="resting",
            targets="treated",
        )


def test_differential_abundance_clr_like_negatives_do_not_crash():
    adata = _tiny_adata(with_negatives=True)
    result = differential_abundance(
        adata,
        contrast_column="condition",
        reference="control",
        targets="treated",
    )

    assert list(result.columns) == EXPECTED_COLUMNS
    assert result["difference"].notna().all()
    planted = result.set_index("marker").loc["M0"]
    assert planted["difference"] > 0
