"""Configuration and shared files/objects for the testing framework.

Copyright © 2022 Pixelgen Technologies AB.
"""

import os
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

from pixelator.config import AntibodyPanel
from pixelator.graph import update_edgelist_membership
from pixelator.graph.utils import union as graph_union
from pixelator.pixeldataset import (
    PixelDataset,
)
from pixelator.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pixeldataset.utils import edgelist_to_anndata
from tests.graph.networkx.test_tools import (
    create_fully_connected_bipartite_graph,
    create_random_graph,
)
from tests.graph.test_graph import (
    create_simple_edge_list_from_graph,
)
from tests.test_tools import enforce_edgelist_types_for_tests

DATA_ROOT = Path(__file__).parent / "data"


def pytest_addoption(parser: pytest.Parser):
    """Register a command line option for pytest.
    This flag is used in workflow integration tests defined in
    tests/integration.
    :param parser: the pytest parser instance
    """
    parser.addoption(
        "--keep-workdirs",
        action="store_true",
        default=False,
        help="Do not delete the working directory.",
    )


@pytest.fixture(name="adata", scope="module")
def adata_fixture(edgelist: pd.DataFrame, panel: AntibodyPanel):
    """Create an anndata instance."""
    adata = edgelist_to_anndata(edgelist=edgelist, panel=panel)
    return adata


@pytest.fixture(name="data_root", scope="session")
def data_root_fixture():
    """Return the data root directory."""
    return DATA_ROOT


@pytest.fixture(name="edgelist", scope="module")
def edgelist_fixture(data_root):
    """Load an example edgelist from disk."""
    edgelist = pl.read_csv(str(data_root / "test_edge_list.csv")).to_pandas()
    g = nx.from_pandas_edgelist(edgelist, source="upia", target="upib")
    node_component_map = pd.Series(index=g.nodes())
    for i, cc in enumerate(nx.connected_components(g)):
        node_component_map[list(cc)] = i
    edgelist = update_edgelist_membership(edgelist, node_component_map)
    return enforce_edgelist_types_for_tests(edgelist)


@pytest.fixture(name="edgelist_with_communities", scope="module")
def edgelist_with_communities_fixture():
    """Create an edgelist that has one component, but two distinct communities."""
    random.seed(7319)

    graph1 = create_fully_connected_bipartite_graph(n_nodes=50)
    graph2 = create_fully_connected_bipartite_graph(n_nodes=50)

    # Make sure to retain the bipartite structure after joining
    source = graph1.vs.select_where(key="bipartite", value=1).get_vertex(50)
    target = graph2.vs.select_where(key="bipartite", value=0).get_vertex(0)

    joined_graph = graph_union([graph1, graph2])
    # Add graph name prefixes to get nodes to match joined graphs
    # with unique node names.
    joined_graph.add_edges([(f"g0-{source.index}", f"g1-{target.index}")])

    def data():
        for upib in joined_graph.vs.select_where(key="bipartite", value=0):
            for upia in upib.neighbors():
                yield {"upia": upia["name"], "upib": upib["name"]}

    edgelist = pd.DataFrame(data())
    edgelist["component"] = "PXLCMP0000000"
    edgelist["umi"] = "UMI"
    edgelist["marker"] = "A"
    edgelist["sequence"] = "ATCG"
    edgelist["count"] = 1

    edgelist = enforce_edgelist_types_for_tests(edgelist)
    return edgelist


@pytest.fixture(name="full_graph_edgelist", scope="module")
def full_graph_edgelist_fixture():
    """Create edgelist from fully connected bipartie graph."""
    random.seed(10)
    g = create_fully_connected_bipartite_graph(50)
    edgelist = create_simple_edge_list_from_graph(g)
    node_component_map = pd.Series(index=g.raw.nodes())
    for i, cc in enumerate(nx.connected_components(g.raw)):
        node_component_map[list(cc)] = i
    edgelist = update_edgelist_membership(edgelist, node_component_map)
    edgelist = enforce_edgelist_types_for_tests(edgelist)
    return edgelist


@pytest.fixture(name="panel", scope="module")
def panel_fixture(data_root):
    """Return a panel."""
    panel = AntibodyPanel.from_csv(str(data_root / "test_panel.csv"))
    return panel


@pytest.fixture(name="pixel_dataset_file")
def pixel_dataset_file(setup_basic_pixel_dataset, tmp_path) -> Path:
    """Create pxl file."""
    dataset, *_ = setup_basic_pixel_dataset
    file_target = tmp_path / "dataset.pxl"
    dataset.save(str(file_target))
    return file_target


@pytest.fixture(name="random_graph_edgelist", scope="module")
def random_graph_edgelist_fixture():
    """Create an edgelist based on a random graph."""
    g = create_random_graph(n_nodes=500, prob=0.005)
    edgelist = create_simple_edge_list_from_graph(g)
    node_component_map = pd.Series(index=g.raw.nodes())
    for i, cc in enumerate(nx.connected_components(g.raw)):
        node_component_map[list(cc)] = i
    edgelist = update_edgelist_membership(edgelist, node_component_map)
    edgelist = enforce_edgelist_types_for_tests(edgelist)
    return edgelist


@pytest.fixture(name="full_random_graph_edgelist", scope="module")
def full_random_graph_edgelist_fixture():
    """Create an edgelist based on a random graph."""
    g = create_random_graph(n_nodes=500, prob=0.005)
    edgelist = create_simple_edge_list_from_graph(g, random_markers=True)
    node_component_map = pd.Series(index=g.raw.nodes())
    for i, cc in enumerate(nx.connected_components(g.raw)):
        node_component_map[list(cc)] = i
    edgelist = update_edgelist_membership(edgelist, node_component_map)
    edgelist = enforce_edgelist_types_for_tests(edgelist)
    return edgelist


@pytest.fixture(name="layout_df")
def layout_df_fixture() -> pd.DataFrame:
    nbr_of_rows = 300
    components = [
        "2ac2ca983a4b82dd",
        "2ac2ca983a4b82dd",
        "6ed5d4e4cfe588bd",
        "701ec72d3bda62d5",
        "bec92437d668cfa1",
        "ce2709afa8ebd1c9",
    ]
    graph_projections = ["bipartite", "a-node"]
    layout_methods = ["pmds", "fr"]
    rgn = np.random.default_rng(1)
    layout_df = pd.DataFrame(
        {
            "x": rgn.random(nbr_of_rows),
            "y": rgn.random(nbr_of_rows),
            "z": rgn.random(nbr_of_rows),
            "graph_projection": rgn.choice(graph_projections, nbr_of_rows),
            "layout": rgn.choice(layout_methods, nbr_of_rows),
            "component": rgn.choice(components, nbr_of_rows),
            "name": "TTTT",
            "pixel_type": "A",
            "index": range(0, nbr_of_rows),
        }
        | {
            marker: rgn.integers(0, 10, nbr_of_rows)
            for marker in ["CD45", "CD3", "CD3", "CD19", "CD19", "CD3"]
        }
    )

    yield layout_df


@pytest.fixture(name="precomputed_layouts")
def precomputed_layouts_fixture(layout_df) -> pd.DataFrame:
    yield PreComputedLayouts(pl.DataFrame(layout_df).lazy())


@pytest.fixture(name="setup_basic_pixel_dataset")
def setup_basic_pixel_dataset(
    edgelist: pd.DataFrame, adata: AnnData, precomputed_layouts: PreComputedLayouts
):
    """Create basic pixel dataset, with some dummy data."""
    # TODO make these dataframes more realistic
    # Right now the edgelist does line up with the polarization
    # and colocalization dataframes, and they do not contain all the
    # columns their real counterparts have.

    polarization_scores = pd.DataFrame(
        data={
            "marker": ["CD45", "CD3", "CD3", "CD19", "CD19", "CD3"],
            "morans_i": [1, 1.5, 0.1, 0.3, 0.1, 1],
            "component": [
                "2ac2ca983a4b82dd",
                "2ac2ca983a4b82dd",
                "6ed5d4e4cfe588bd",
                "701ec72d3bda62d5",
                "bec92437d668cfa1",
                "ce2709afa8ebd1c9",
            ],
        },
    )
    colocalization_scores = pd.DataFrame(
        data={
            "marker_1": ["CD45", "CD3", "CD19", "CD19", "CD45"],
            "marker_2": ["CD3", "CD19", "CD45", "CD45", "CD19"],
            "pearson": [0.1, 0.5, 0.3, 0.2, 0.1],
            "component": [
                "2ac2ca983a4b82dd",
                "6ed5d4e4cfe588bd",
                "701ec72d3bda62d5",
                "bec92437d668cfa1",
                "ce2709afa8ebd1c9",
            ],
        },
    )

    metadata = {"A": 1, "B": 2, "file_format_version": 1}

    dataset = PixelDataset.from_data(
        edgelist=edgelist,
        adata=adata,
        metadata=metadata,
        polarization=polarization_scores,
        colocalization=colocalization_scores,
        precomputed_layouts=precomputed_layouts,
    )
    return (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
        precomputed_layouts,
    )


@pytest.fixture
def enable_backend(request):
    previous_environment = os.environ
    if request.param == "networkx":
        new_environment = previous_environment.copy()
        new_environment["PIXELATOR_GRAPH_BACKEND"] = "NetworkXGraphBackend"
        os.environ = new_environment
    yield
    os.environ = previous_environment
