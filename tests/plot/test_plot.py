"""Tests for the plot module.

Copyright © 2023 Pixelgen Technologies AB.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_almost_equal
from pytest_snapshot.plugin import Snapshot

from pixelator.graph import Graph
from pixelator.plot import (
    abundance_colocalization_plot,
    cell_count_plot,
    density_scatter_plot,
    edge_rank_plot,
    molecule_rank_plot,
    plot_2d_graph,
    plot_3d_graph,
    plot_3d_heatmap,
    plot_colocalization_diff_heatmap,
    plot_colocalization_diff_volcano,
    plot_colocalization_heatmap,
    plot_polarity_diff_volcano,
    scatter_umi_per_upia_vs_tau,
)
from pixelator.plot.layout_plots import (
    _calculate_densities,
    _calculate_distance_to_unit_sphere_zones,
    _unit_sphere_surface,
)


@pytest.mark.parametrize(
    "component, marker",
    [
        ("2ac2ca983a4b82dd", "CD45RA"),
        ("2ac2ca983a4b82dd", None),
    ],
)
def test_plot_3d_graph(
    snapshot: Snapshot, component, marker, setup_basic_pixel_dataset
):
    """Test `plot_3d_graph` function.

    :param snapshot: testing snapshot directory
    """
    np.random.seed(0)
    snapshot.snapshot_dir = "tests/snapshots/test_plot/test_plot_3d_graph"
    pxl_data, *_ = setup_basic_pixel_dataset
    result = plot_3d_graph(
        pxl_data,
        layout_algorithm="pmds_3d",
        component=component,
        marker=marker,
        suppress_fig=True,
    )
    assert isinstance(result, go.Figure)
    # snapshot.assert_match(result.to_json(), "plot_3d_graph_fig.json")
    # TODO: Fix the snapshot test - Even though the plotly version matches (5.18.0), the test fails on github


@pytest.mark.parametrize(
    "component, marker",
    [
        ("2ac2ca983a4b82dd", "CD45"),
        ("2ac2ca983a4b82dd", None),
    ],
)
def test_plot_3d_graph_precomputed(
    snapshot: Snapshot, component, marker, setup_basic_pixel_dataset
):
    """Test `plot_3d_graph` function.

    :param snapshot: testing snapshot directory
    """
    np.random.seed(0)
    snapshot.snapshot_dir = "tests/snapshots/test_plot/test_plot_3d_graph"
    pxl_data, *_ = setup_basic_pixel_dataset
    assert pxl_data.precomputed_layouts is not None
    result = plot_3d_graph(
        pxl_data,
        layout_algorithm=None,
        component=component,
        marker=marker,
        suppress_fig=True,
    )
    assert isinstance(result, go.Figure)


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_2d_graph_precomputed",
)
@pytest.mark.parametrize(
    "component, marker, show_b_nodes",
    [
        (("2ac2ca983a4b82dd", "CD45", False)),
        ((["6ed5d4e4cfe588bd", "701ec72d3bda62d5"], ["CD3", "CD45", "CD19"], False)),
        (("2ac2ca983a4b82dd", "pixel_type", True)),
    ],
)
def test_plot_2d_graph_precomputed(
    setup_basic_pixel_dataset, component, marker, show_b_nodes
):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    assert pxl_data.precomputed_layouts is not None
    fig, _ = plot_2d_graph(
        pxl_data,
        layout_algorithm=None,
        component=component,
        marker=marker,
        show_b_nodes=show_b_nodes,
        random_seed=0,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_2d_graph",
)
@pytest.mark.parametrize(
    "component, marker, show_b_nodes",
    [
        ("2ac2ca983a4b82dd", "CD45RA", False),
        ((["6ed5d4e4cfe588bd", "701ec72d3bda62d5"], ["CD20", "CD45", "CD45RA"], False)),
        (("2ac2ca983a4b82dd", "pixel_type", True)),
    ],
)
def test_plot_2d_graph(setup_basic_pixel_dataset, component, marker, show_b_nodes):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    fig, _ = plot_2d_graph(
        pxl_data,
        layout_algorithm="pmds",
        component=component,
        marker=marker,
        show_b_nodes=show_b_nodes,
        random_seed=0,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_colocalization_heatmap",
)
def test_plot_colocalization_heatmap(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    fig, _ = plot_colocalization_heatmap(
        pxl_data.colocalization, value_column="pearson"
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_colocalization_diff_heatmap",
)
def test_plot_colocalization_diff_heatmap(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    colocalization_data = pxl_data.colocalization
    colocalization_data.loc[5] = [
        "CD3",
        "CD19",
        0.5,
        "701ec72d3bda62d5",
    ]  # Adding a new pair of colocalization data as the heatmap needs at least 2 rows
    colocalization_data.loc[6] = ["CD3", "CD19", 0.7, "ce2709afa8ebd1c9"]
    fig, _ = plot_colocalization_diff_heatmap(
        colocalization_data,
        targets="ce2709afa8ebd1c9",
        reference="701ec72d3bda62d5",
        contrast_column="component",
        value_column="pearson",
        min_log_p=0,
    )
    return fig["ce2709afa8ebd1c9"]


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_colocalization_diff_volcano",
)
def test_plot_colocalization_diff_volcano(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    colocalization_data = pxl_data.colocalization
    colocalization_data.loc[5] = [
        "CD3",
        "CD19",
        0.5,
        "701ec72d3bda62d5",
    ]  # Adding a new pair of colocalization data as the volcano needs at least 2 rows
    colocalization_data.loc[6] = ["CD3", "CD19", 0.7, "ce2709afa8ebd1c9"]
    fig, _ = plot_colocalization_diff_volcano(
        colocalization_data,
        targets="ce2709afa8ebd1c9",
        reference="701ec72d3bda62d5",
        contrast_column="component",
        value_column="pearson",
        min_log_p=-1,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_polarity_diff_volcano",
)
def test_plot_polarity_diff_volcano(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    polarity_data = pxl_data.polarization
    fig, _ = plot_polarity_diff_volcano(
        polarity_data,
        targets="ce2709afa8ebd1c9",
        reference="701ec72d3bda62d5",
        contrast_column="component",
        value_column="morans_i",
        min_log_p=-1,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_colocalization_diff_volcano_multiple",
)
def test_plot_colocalization_diff_volcano_multiple(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    colocalization_data = pxl_data.colocalization
    colocalization_data.loc[5] = [
        "CD3",
        "CD19",
        0.5,
        "701ec72d3bda62d5",
    ]  # Adding a new pair of colocalization data as the volcano needs at least 2 rows
    colocalization_data.loc[6] = ["CD3", "CD19", 0.7, "ce2709afa8ebd1c9"]
    fig, _ = plot_colocalization_diff_volcano(
        colocalization_data,
        reference="701ec72d3bda62d5",
        contrast_column="component",
        value_column="pearson",
        min_log_p=-1,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_plot_polarity_diff_volcano_multiple",
)
def test_plot_polarity_diff_volcano_multiple(setup_basic_pixel_dataset):
    np.random.seed(0)
    pxl_data, *_ = setup_basic_pixel_dataset
    polarity_data = pxl_data.polarization
    fig, _ = plot_polarity_diff_volcano(
        polarity_data,
        reference="701ec72d3bda62d5",
        contrast_column="component",
        value_column="morans_i",
        min_log_p=-1,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="../snapshots/test_plot/test_scatter_umi_per_upia_vs_tau",
)
def test_scatter_umi_per_upia_vs_tau():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "umi_per_upia": np.random.uniform(1, 10, 100),
            "tau": np.random.uniform(0.9, 1, 100),
            "tau_type": np.random.choice(["high", "low", "normal"], 100),
            "group": np.random.choice(["A", "B"], 100),
        }
    )
    plot, _ = scatter_umi_per_upia_vs_tau(data, group_by="group")
    return plot


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="../snapshots/test_plot/test_cell_count_plot",
)
def test_cell_count_plot():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "group1": np.random.choice(["A", "B"], 100),
            "group2": np.random.choice(["C", "D"], 100),
        }
    )
    plot, _ = cell_count_plot(data, color_by="group1", group_by="group2")
    return plot


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="../snapshots/test_plot/test_molecule_rank_plot",
)
def test_molecule_rank_plot():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "molecules": np.round(10 ** np.random.normal(4, 0.3, 500)).astype(int),
            "group": np.random.choice(["A", "B"], 500),
        }
    )
    plot, _ = molecule_rank_plot(data, group_by="group")
    return plot


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="../snapshots/test_plot/test_molecule_rank_plot",
)
def test_molecule_rank_plot_back_compatibility():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "edges": np.round(10 ** np.random.normal(4, 0.3, 500)).astype(int),
            "group": np.random.choice(["A", "B"], 500),
        }
    )
    plot, _ = molecule_rank_plot(data, group_by="group")
    return plot


@pytest.mark.mpl_image_compare(
    deterministic=False,
    baseline_dir="../snapshots/test_plot/test_edge_rank_plot",
)
def test_edge_rank_plot():
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "edges": np.round(10 ** np.random.normal(4, 0.3, 500)).astype(int),
            "group": np.random.choice(["A", "B"], 500),
        }
    )
    plot, _ = edge_rank_plot(data, group_by="group")
    return plot


def test__calculate_distance_to_unit_sphere_zones():
    rng = np.random.default_rng(seed=10)
    sphere_points = rng.standard_normal((10, 3))
    sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1)[:, None]

    unit_sphere = _unit_sphere_surface(horizontal_resolution=5, vertical_resolution=5)
    result = _calculate_distance_to_unit_sphere_zones(
        sphere_points, unit_sphere_surface=unit_sphere
    )
    assert_almost_equal(
        result,
        np.array(
            [
                [
                    1.73754683,
                    1.1596561,
                    1.12478892,
                    1.83287282,
                    1.97058844,
                    1.92060464,
                    1.95228706,
                    1.06465498,
                    1.56226933,
                    1.0887463,
                ],
                [
                    1.93327148,
                    0.74752888,
                    0.74733425,
                    1.86896772,
                    1.95002159,
                    1.98134057,
                    1.87377263,
                    1.56192074,
                    1.55150757,
                    0.70497397,
                ],
                [
                    1.85423345,
                    0.78551405,
                    0.83442976,
                    1.65893191,
                    1.63279522,
                    1.74202288,
                    1.52495431,
                    1.86767047,
                    1.46121457,
                    0.83009153,
                ],
                [
                    1.51537369,
                    1.21876533,
                    1.26401865,
                    1.25320636,
                    1.06738495,
                    1.23995834,
                    0.97431123,
                    1.91442584,
                    1.33564681,
                    1.28415388,
                ],
                [
                    0.99041961,
                    1.62947774,
                    1.65373815,
                    0.80036068,
                    0.34173267,
                    0.55792278,
                    0.43425248,
                    1.69307702,
                    1.24872517,
                    1.67768635,
                ],
                [
                    1.73754683,
                    1.1596561,
                    1.12478892,
                    1.83287282,
                    1.97058844,
                    1.92060464,
                    1.95228706,
                    1.06465498,
                    1.56226933,
                    1.0887463,
                ],
                [
                    1.84088198,
                    1.56436299,
                    0.73635711,
                    1.96261815,
                    1.84445318,
                    1.73117445,
                    1.65146509,
                    1.45136694,
                    1.92035238,
                    0.72428063,
                ],
                [
                    1.71611669,
                    1.81319571,
                    0.82051214,
                    1.80541322,
                    1.44897179,
                    1.31194483,
                    1.10326107,
                    1.73697307,
                    1.98649392,
                    0.85327368,
                ],
                [
                    1.39558778,
                    1.83679641,
                    1.25755972,
                    1.38901255,
                    0.85949635,
                    0.7802251,
                    0.4069342,
                    1.82534816,
                    1.75057988,
                    1.29485341,
                ],
                [
                    0.99041961,
                    1.62947774,
                    1.65373815,
                    0.80036068,
                    0.34173267,
                    0.55792278,
                    0.43425248,
                    1.69307702,
                    1.24872517,
                    1.67768635,
                ],
                [
                    1.73754683,
                    1.1596561,
                    1.12478892,
                    1.83287282,
                    1.97058844,
                    1.92060464,
                    1.95228706,
                    1.06465498,
                    1.56226933,
                    1.0887463,
                ],
                [
                    1.3052366,
                    1.58575253,
                    1.54992156,
                    1.55867694,
                    1.69135725,
                    1.56923654,
                    1.74663037,
                    0.57876914,
                    1.48863951,
                    1.5332804,
                ],
                [
                    0.7495454,
                    1.83928456,
                    1.81761574,
                    1.11711454,
                    1.15498042,
                    0.98252547,
                    1.29403027,
                    0.71540689,
                    1.36559584,
                    1.81960107,
                ],
                [
                    0.51230984,
                    1.85504732,
                    1.85512574,
                    0.71201101,
                    0.444315,
                    0.27256106,
                    0.69926829,
                    1.24916116,
                    1.26207141,
                    1.87163343,
                ],
                [
                    0.99041961,
                    1.62947774,
                    1.65373815,
                    0.80036068,
                    0.34173267,
                    0.55792278,
                    0.43425248,
                    1.69307702,
                    1.24872517,
                    1.67768635,
                ],
                [
                    1.73754683,
                    1.1596561,
                    1.12478892,
                    1.83287282,
                    1.97058844,
                    1.92060464,
                    1.95228706,
                    1.06465498,
                    1.56226933,
                    1.0887463,
                ],
                [
                    1.4325972,
                    0.79131469,
                    1.55516672,
                    1.43897329,
                    1.80589757,
                    1.84153436,
                    1.95816357,
                    0.81737636,
                    0.96719702,
                    1.52425544,
                ],
                [
                    1.02710443,
                    0.84399131,
                    1.82394074,
                    0.86051327,
                    1.37857925,
                    1.50956972,
                    1.66817715,
                    0.99142552,
                    0.23203858,
                    1.80884605,
                ],
                [
                    0.7817631,
                    1.24610129,
                    1.85951021,
                    0.38487659,
                    0.77329974,
                    1.00151635,
                    1.12812369,
                    1.37605741,
                    0.55879043,
                    1.86424718,
                ],
                [
                    0.99041961,
                    1.62947774,
                    1.65373815,
                    0.80036068,
                    0.34173267,
                    0.55792278,
                    0.43425248,
                    1.69307702,
                    1.24872517,
                    1.67768635,
                ],
                [
                    1.73754683,
                    1.1596561,
                    1.12478892,
                    1.83287282,
                    1.97058844,
                    1.92060464,
                    1.95228706,
                    1.06465498,
                    1.56226933,
                    1.0887463,
                ],
                [
                    1.93327148,
                    0.74752888,
                    0.74733425,
                    1.86896772,
                    1.95002159,
                    1.98134057,
                    1.87377263,
                    1.56192074,
                    1.55150757,
                    0.70497397,
                ],
                [
                    1.85423345,
                    0.78551405,
                    0.83442976,
                    1.65893191,
                    1.63279522,
                    1.74202288,
                    1.52495431,
                    1.86767047,
                    1.46121457,
                    0.83009153,
                ],
                [
                    1.51537369,
                    1.21876533,
                    1.26401865,
                    1.25320636,
                    1.06738495,
                    1.23995834,
                    0.97431123,
                    1.91442584,
                    1.33564681,
                    1.28415388,
                ],
                [
                    0.99041961,
                    1.62947774,
                    1.65373815,
                    0.80036068,
                    0.34173267,
                    0.55792278,
                    0.43425248,
                    1.69307702,
                    1.24872517,
                    1.67768635,
                ],
            ]
        ),
    )


def test__calculate_densities():
    rng = np.random.default_rng(seed=10)
    sphere_points = rng.standard_normal((100, 3))
    sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1)[:, None]

    unit_sphere = _unit_sphere_surface(horizontal_resolution=10, vertical_resolution=10)

    result = _calculate_densities(
        sphere_points, distance_cutoff=0.4, unit_sphere_surface=unit_sphere
    )
    assert_almost_equal(
        result,
        np.array(
            [
                0.14527102,
                0.0,
                0.22485373,
                0.38391647,
                0.52441957,
                0.0,
                0.0,
                0.64520005,
                0.71133096,
                0.93961753,
                0.14527102,
                0.56926217,
                0.45965503,
                0.22779889,
                0.0,
                0.0,
                0.16770936,
                0.18212749,
                0.60333476,
                0.93961753,
                0.14527102,
                0.66695579,
                0.0,
                0.0,
                0.0,
                0.0578167,
                0.0,
                0.2797887,
                0.75478722,
                0.93961753,
                0.14527102,
                0.40079531,
                0.0,
                0.0,
                0.0,
                0.0,
                0.14025073,
                0.46362265,
                0.95052605,
                0.93961753,
                0.14527102,
                0.31687737,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.34143256,
                1.0,
                0.93961753,
                0.14527102,
                0.37872688,
                0.49108485,
                0.34803694,
                0.37297797,
                0.64201735,
                0.54968217,
                0.33573108,
                0.94184454,
                0.93961753,
                0.14527102,
                0.0,
                0.35518955,
                0.0,
                0.0,
                0.69668229,
                0.62506557,
                0.83941926,
                0.70312585,
                0.93961753,
                0.14527102,
                0.0,
                0.25168217,
                0.0,
                0.0,
                0.31160573,
                0.0,
                0.6030988,
                0.59564422,
                0.93961753,
                0.14527102,
                0.0,
                0.19712799,
                0.38605462,
                0.0,
                0.0,
                0.45301558,
                0.95876429,
                0.79786941,
                0.93961753,
                0.14527102,
                0.0,
                0.22485373,
                0.38391647,
                0.52441957,
                0.0,
                0.0,
                0.64520005,
                0.71133096,
                0.93961753,
            ]
        ),
    )


def test_plot_3d_heatmap(edgelist):
    component_1 = edgelist["component"].unique()[0]
    graph = Graph.from_edgelist(
        edgelist[edgelist["component"] == component_1],
        add_marker_counts=True,
        simplify=True,
        use_full_bipartite=False,
    )
    with pytest.warns(
        UserWarning,
        match=(
            "The function `plot_3d_heatmap` is experimental, "
            "it might be removed or the API might change without notice."
        ),
    ):
        # For now, just making sure that this doesn't crash when running.
        _ = plot_3d_heatmap(
            graph,
            layout_algorithm="fruchterman_reingold_3d",
            marker="CD3",
            distance_cutoff=0.4,
        )


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_density_scatter_plot/",
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
    setup_basic_pixel_dataset, marker1, marker2, extra_params
):
    facet_row = extra_params["facet_row"]
    facet_column = extra_params["facet_column"]
    gate = extra_params["gate"]

    pxl_data, *_ = setup_basic_pixel_dataset
    np.random.seed(0)
    pxl_data.adata[:, marker1] = pxl_data.adata[
        :, marker1
    ].X.flatten() + np.random.randint(1, 20, size=pxl_data.adata.shape[0])
    pxl_data.adata[:, marker2] = pxl_data.adata[
        :, marker2
    ].X.flatten() + np.random.randint(1, 20, size=pxl_data.adata.shape[0])
    show_marginal = (facet_column is None) & (facet_row is None)
    fig, _ = density_scatter_plot(
        pxl_data.adata,
        marker1=marker1,
        marker2=marker2,
        facet_row=facet_row,
        facet_column=facet_column,
        gate=gate,
        show_marginal=show_marginal,
    )
    return fig


@pytest.mark.mpl_image_compare(
    deterministic=True,
    baseline_dir="../snapshots/test_plot/test_abundance_colocalization_plot/",
)
def test_abundance_colocalization_plot(setup_basic_pixel_dataset):
    pixel_data, *_ = setup_basic_pixel_dataset
    fig, _ = abundance_colocalization_plot(
        pixel_data,
        markers_x=["CD3", "CD8"],
        markers_y=["CD14", "CD19"],
        colocalization_column="pearson",
    )
    return fig
