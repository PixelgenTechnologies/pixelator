"""Configuration and shared files/objects for the testing framework.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import random
from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData
from pixelator.config import AntibodyPanel
from pixelator.graph import Graph, update_edgelist_membership
from pixelator.pixeldataset import (
    PixelDataset,
    edgelist_to_anndata,
)

from tests.graph.igraph.test_tools import (
    create_fully_connected_bipartite_graph,
    create_random_graph,
)
from tests.graph.test_graph import (
    create_simple_edge_list_from_graph,
)

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
    edgelist = pd.read_csv(
        str(data_root / "test_edge_list.csv"), dtype_backend="pyarrow"
    )
    edgelist = update_edgelist_membership(edgelist, prefix="PXLCMP")
    return edgelist


@pytest.fixture(name="edgelist_with_communities", scope="module")
def edgelist_with_communities_fixture():
    """Create an edgelist that has to communities."""
    random.seed(7319)

    graph1 = create_fully_connected_bipartite_graph(n_nodes=50)
    graph2 = create_fully_connected_bipartite_graph(n_nodes=50)

    # Make sure to retain the bipartite structure after joining
    source = graph1.vs.select(type=True)[0]["name"]
    target = graph2.vs.select(type=False)[0]["name"]
    joined_graph = Graph.union([graph1, graph2])
    joined_graph.add_edges([(source, target)])

    def data():
        for upib in joined_graph.vs.select(type=True):
            for upia in upib.neighbors():
                yield {"upia": upia["name"], "upib": upib["name"]}

    edgelist = pd.DataFrame(data())
    edgelist["component"] = "PXLCMP0000000"
    edgelist["umi"] = "UMI"
    edgelist["marker"] = "A"
    edgelist["sequence"] = "ATCG"
    return edgelist


@pytest.fixture(name="full_graph_edgelist", scope="module")
def full_graph_edgelist_fixture():
    """Create edgelist from fully connected bipartie graph."""
    g = create_fully_connected_bipartite_graph(50)
    edgelist = create_simple_edge_list_from_graph(g)
    edgelist = update_edgelist_membership(edgelist, prefix="PXLCMP")
    return edgelist


@pytest.fixture(name="panel", scope="module")
def panel_fixture(data_root):
    """Return a panel."""
    panel = AntibodyPanel.from_csv(str(data_root / "test_panel.csv"))
    return panel


@pytest.fixture(name="pixel_dataset_file")
def pixel_dataset_file(setup_basic_pixel_dataset, tmp_path):
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
    edgelist = update_edgelist_membership(edgelist, prefix="PXLCMP")
    return edgelist


@pytest.fixture(name="setup_basic_pixel_dataset")
def setup_basic_pixel_dataset(edgelist: pd.DataFrame, adata: AnnData):
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
                "PXLCMP0000000",
                "PXLCMP0000000",
                "PXLCMP0000001",
                "PXLCMP0000002",
                "PXLCMP0000003",
                "PXLCMP0000004",
            ],
        },
    )
    colocalization_scores = pd.DataFrame(
        data={
            "marker_1": ["CD45", "CD3", "CD19", "CD19", "CD45"],
            "marker_2": ["CD3", "CD19", "CD45", "CD45", "CD19"],
            "pearson": [0.1, 0.5, 0.3, 0.2, 0.1],
            "component": [
                "PXLCMP0000000",
                "PXLCMP0000001",
                "PXLCMP0000002",
                "PXLCMP0000003",
                "PXLCMP0000004",
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
    )
    return (
        dataset,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    )
