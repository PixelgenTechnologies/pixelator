"""Tests for the top-level plot package.

Copyright © 2023 Pixelgen Technologies AB.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy.stats import gaussian_kde

from pixelator.plot import _estimate_point_density, density_scatter_plot

DATA_ROOT = Path(__file__).parent / "data"


@pytest.fixture(name="density_scatter_plot_adata")
def density_scatter_plot_adata_fixture() -> AnnData:
    """Return AnnData used by the density scatter plot image-compare test."""
    return read_h5ad(DATA_ROOT / "density_scatter_plot_input.h5ad")


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="snapshots/test_density_scatter_plot/",
)
@pytest.mark.parametrize(
    "marker1, marker2, extra_params",
    [
        (
            "CD3",
            "CD8",
            {
                "facet_row": None,
                "facet_column": None,
                "gate": pd.Series(
                    [600, 10, 1000, 20], index=["xmin", "ymin", "xmax", "ymax"]
                ),
            },
        ),
        (
            "CD3",
            "CD8",
            {
                "facet_row": "mean_molecules_per_a_pixel",
                "facet_column": None,
                "gate": None,
            },
        ),
        (
            "CD3",
            "CD8",
            {
                "facet_row": None,
                "facet_column": "mean_molecules_per_a_pixel",
                "gate": pd.Series(
                    [600, 10, 1000, 20], index=["xmin", "ymin", "xmax", "ymax"]
                ),
            },
        ),
    ],
)
def test_density_scatter_plot(
    marker1, marker2, extra_params, density_scatter_plot_adata
):
    """Verify density scatter plot.

    Using the old fixture-derived AnnData.

    Args:
        marker1: marker1.
        marker2: marker2.
        extra_params: extra params.
        density_scatter_plot_adata: AnnData fixture for the plot input.
    """

    facet_row = extra_params["facet_row"]
    facet_column = extra_params["facet_column"]
    gate = extra_params["gate"]

    show_marginal = (facet_column is None) & (facet_row is None)
    fig, _ = density_scatter_plot(
        density_scatter_plot_adata,
        marker1=marker1,
        marker2=marker2,
        facet_row=facet_row,
        facet_column=facet_column,
        gate=gate,
        show_marginal=show_marginal,
    )
    return fig


def _separated_facets() -> tuple[np.ndarray, pd.DataFrame, int]:
    """Two clouds. The co-culture cloud is bimodal; cart sits on one of the modes."""
    rng = np.random.default_rng(0)
    cart = rng.normal(loc=(0.0, 0.0), scale=0.4, size=(400, 2))
    coculture_near_cart = rng.normal(loc=(0.0, 0.0), scale=0.4, size=(30, 2))
    coculture_own_mode = rng.normal(loc=(10.0, 10.0), scale=0.4, size=(80, 2))
    points = np.vstack([cart, coculture_near_cart, coculture_own_mode])
    labels = ["cart"] * len(cart) + ["coculture"] * (
        len(coculture_near_cart) + len(coculture_own_mode)
    )
    return points, pd.DataFrame({"condition": labels}), len(cart)


def _closest_center(point: np.ndarray) -> str:
    cart = np.array([0.0, 0.0])
    own = np.array([10.0, 10.0])
    if np.linalg.norm(point - own) < np.linalg.norm(point - cart):
        return "coculture"
    return "cart"


def test_per_facet_density_peaks_at_each_facets_own_mode():
    """Each facet peaks at 1 on its own mode, not on a mode borrowed from another facet."""
    points, groups, cart_n = _separated_facets()

    density = _estimate_point_density(points, groups, per_group=True, scale_to_max=True)

    cart_peak = points[:cart_n][np.argmax(density[:cart_n])]
    coculture = density[cart_n:]
    coculture_peak = points[cart_n:][np.argmax(coculture)]
    assert density[:cart_n].max() == pytest.approx(1.0)
    assert coculture.max() == pytest.approx(1.0)
    assert _closest_center(cart_peak) == "cart"
    assert _closest_center(coculture_peak) == "coculture"


def test_joint_density_matches_one_kde_on_all_points():
    """``density_scope='joint'`` is one kernel estimate on the whole dataset."""
    points, groups, cart_n = _separated_facets()

    joint = _estimate_point_density(points, groups, per_group=False, scale_to_max=False)
    expected = gaussian_kde(points.T)(points.T)

    np.testing.assert_allclose(joint, expected)
    coculture_peak = points[cart_n:][np.argmax(joint[cart_n:])]
    assert _closest_center(coculture_peak) == "cart"


def test_facet_combinations_are_scaled_separately():
    """Row and column together define a facet, and each combination peaks at 1."""
    rng = np.random.default_rng(1)
    left = rng.normal(loc=(0.0, 0.0), scale=0.3, size=(40, 2))
    right = rng.normal(loc=(12.0, 0.0), scale=0.3, size=(200, 2))
    points = np.vstack([left, right])
    groups = pd.DataFrame(
        {
            "row": ["donor"] * len(points),
            "column": ["a"] * len(left) + ["b"] * len(right),
        }
    )

    density = _estimate_point_density(points, groups, per_group=True, scale_to_max=True)

    assert density[: len(left)].max() == pytest.approx(1.0)
    assert density[len(left) :].max() == pytest.approx(1.0)


def test_scale_to_max_is_optional_for_a_single_density():
    """Non-faceted density stays unscaled unless max scaling is requested."""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(50, 2))

    raw = _estimate_point_density(points, scale_to_max=False)
    scaled = _estimate_point_density(points, scale_to_max=True)

    np.testing.assert_allclose(raw, gaussian_kde(points.T)(points.T))
    assert raw.max() != pytest.approx(1.0)
    assert scaled.max() == pytest.approx(1.0)
    np.testing.assert_allclose(scaled, raw / raw.max())


def test_small_or_flat_facets_get_a_constant_density():
    """One point, or points with no area, do not crash the kernel estimate."""
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [0.0, 5.0],
            [3.0, 5.0],
            [6.0, 5.0],
        ]
    )
    groups = pd.DataFrame(
        {"sample": ["single", "identical", "identical", "line", "line", "line"]}
    )

    density = _estimate_point_density(points, groups, per_group=True, scale_to_max=True)

    assert density.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_joint_density_still_raises_when_the_kernel_cannot_be_estimated():
    """The constant-density fallback is per facet, not for one joint estimate."""
    collinear = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        _estimate_point_density(collinear, per_group=False)

    one_point = np.array([[0.0, 0.0]])
    with pytest.raises(ValueError):
        _estimate_point_density(one_point, per_group=False)

    adata = AnnData(
        X=np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        obs=pd.DataFrame({"sample": ["a", "a", "b", "b"]}),
        var=pd.DataFrame(index=["m1", "m2"]),
    )
    fig, _ = density_scatter_plot(adata, "m1", "m2", facet_column="sample")
    plt.close(fig)
    with pytest.raises(np.linalg.LinAlgError):
        density_scatter_plot(
            adata, "m1", "m2", facet_column="sample", density_scope="joint"
        )
    with pytest.raises(np.linalg.LinAlgError):
        density_scatter_plot(adata, "m1", "m2")


def test_density_scatter_plot_draws_a_facet_with_too_few_points():
    """A panel with one cell, and a flat panel, still produce a figure."""
    values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 4.0],
            [2.0, 4.1],
            [4.0, 3.8],
        ]
    )
    adata = AnnData(
        X=values,
        obs=pd.DataFrame({"sample": ["single", "flat", "flat", "ok", "ok", "ok"]}),
        var=pd.DataFrame(index=["m1", "m2"]),
    )

    fig, _ = density_scatter_plot(adata, "m1", "m2", facet_column="sample")

    assert len(fig.axes) == 3
    plt.close(fig)


def test_density_scatter_plot_rejects_an_unknown_density_scope():
    adata = AnnData(X=np.ones((2, 2)), var=pd.DataFrame(index=["m1", "m2"]))

    with pytest.raises(ValueError, match="density_scope"):
        density_scatter_plot(adata, "m1", "m2", density_scope="panel")


def test_density_scope_selects_per_facet_or_joint_estimation(monkeypatch):
    """The plot keyword chooses a per-facet estimate or one joint estimate."""
    seen = {}

    def record(points, groups=None, *, per_group=False, scale_to_max=False):
        seen["per_group"] = per_group
        seen["scale_to_max"] = scale_to_max
        seen["n_groups"] = None if groups is None else groups["sample"].nunique()
        return np.ones(len(points))

    monkeypatch.setattr("pixelator.plot._estimate_point_density", record)
    values = np.array([[0.0, 0.0], [1.0, 1.0], [8.0, 8.0], [9.0, 8.5]])
    adata = AnnData(
        X=values,
        obs=pd.DataFrame({"sample": ["a", "a", "b", "b"]}),
        var=pd.DataFrame(index=["m1", "m2"]),
    )

    fig, _ = density_scatter_plot(adata, "m1", "m2", facet_column="sample")
    plt.close(fig)
    assert seen == {"per_group": True, "scale_to_max": True, "n_groups": 2}

    fig, _ = density_scatter_plot(
        adata, "m1", "m2", facet_column="sample", density_scope="joint"
    )
    plt.close(fig)
    assert seen["per_group"] is False
    assert seen["scale_to_max"] is False
    assert seen["n_groups"] is None
