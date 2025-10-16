"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from unittest.mock import create_autospec

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config import PNAAntibodyPanel, load_antibody_panel
from pixelator.pna.pixeldataset.io import PixelFileWriter

mock_antibody_panel = create_autospec(PNAAntibodyPanel)
mock_antibody_panel.markers = ["A", "B", "C"]
# Each marker is duplicated in the panel,
# on these parameters so this accounts for that.
mock_antibody_panel.df = pd.DataFrame(
    {
        "marker_id": [
            "A",
            "B",
            "C",
        ],
        "uniprot_id": ["P61769", "P05107", "P15391"],
        "control": ["no", "no", "yes"],
        "nuclear": ["yes", "no", "no"],
        "sequence_1": ["AAAA", "CCCC", "GGGG"],
    }
)


@pytest.fixture(name="edgelist")
def create_edgelist():
    # Please note that the graph has a bipartite structure
    # i.e. a node cannot be both a umi1 and a umi2
    node_to_maker_marker_map = {
        # component 1
        1: "A",  # 1
        2: "A",
        3: "C",
        4: "B",  # 1
        5: "B",
        6: "A",
        7: "A",  # 1
        # component 2
        8: "C",
        9: "B",
        10: "A",
        11: "C",
        12: "C",
        13: "B",
        # component 3
        14: "A",
        15: "B",
        16: "A",
    }
    # Please note that the graph has a bipartite structure
    # i.e. a node cannot be both a umi1 and a umi2
    data = {
        "umi1": [
            1,
            1,
            4,
            4,
            7,
            7,
            7,
            8,
            8,
            11,
            11,
            11,
            13,
            14,
            15,
        ],
        "umi2": [
            2,
            3,
            2,
            5,
            2,
            5,
            6,
            9,
            10,
            9,
            10,
            12,
            12,
            16,
            16,
        ],
        "component": [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "3",
            "3",
        ],
        "read_count": [
            1,
            2,
            1,
            4,
            1,
            2,
            1,
            2,
            5,
            2,
            2,
            2,
            1,
            3,
            1,
        ],
    }
    df = pd.DataFrame(data)
    df["marker_1"] = [node_to_maker_marker_map[node] for node in df["umi1"]]
    df["marker_2"] = [node_to_maker_marker_map[node] for node in df["umi2"]]

    return pl.LazyFrame(df)


@pytest.fixture(name="pixelconnection")
def create_pixel_dataset_connection(edgelist):
    from pixelator import __version__

    with tempfile.TemporaryDirectory() as temp_dir:
        with PixelFileWriter(temp_dir + "/tmp.pxl") as writer:
            writer.write_metadata(
                {
                    "sample_name": "temp",
                    "version": __version__,
                    "technology": "single-cell-pna",
                    "panel_name": "mock.name",
                    "panel_version": "mock.version",
                }
            )
            writer.write_edgelist(edgelist.collect())
            yield writer.get_connection()


def test_pna_edgelist_to_anndata(pixelconnection):
    adata = pna_edgelist_to_anndata(pixelconnection, mock_antibody_panel)

    assert adata.shape == (3, 3)

    expected_df = pd.DataFrame.from_dict(
        {
            "1": {"A": 4, "B": 2, "C": 1},
            "2": {"A": 1, "B": 2, "C": 3},
            "3": {"A": 2, "B": 1, "C": 0},
        },
        orient="index",
    ).sort_index()
    expected_df.index.name = "component"
    expected_df = expected_df.astype(np.uint32)
    assert_frame_equal(adata.to_df().sort_index(), expected_df)

    expected_var = pd.DataFrame.from_dict(
        {
            "A": {
                "antibody_count": 7,
                "antibody_pct": np.float32(0.4375),
                "components": 3,
                "uniprot_id": "P61769",
                "control": "no",
                "nuclear": "yes",
                "sequence_1": "AAAA",
            },
            "B": {
                "antibody_count": 5,
                "antibody_pct": np.float32(0.3125),
                "components": 3,
                "uniprot_id": "P05107",
                "control": "no",
                "nuclear": "no",
                "sequence_1": "CCCC",
            },
            "C": {
                "antibody_count": 4,
                "antibody_pct": np.float32(0.25),
                "components": 2,
                "uniprot_id": "P15391",
                "control": "yes",
                "nuclear": "no",
                "sequence_1": "GGGG",
            },
        },
        orient="index",
    )
    assert_frame_equal(adata.var, expected_var)

    expected_obs = pd.DataFrame.from_dict(
        {
            "1": {
                "n_umi": 7,
                "n_umi1": 3,
                "n_umi2": 4,
                "n_edges": 7,
                "n_antibodies": 3,
                "reads_in_component": 12,
                "isotype_fraction": 0.14285714285714285,
                "intracellular_fraction": 0.5714285714285714,
            },
            "2": {
                "n_umi": 6,
                "n_umi1": 3,
                "n_umi2": 3,
                "n_edges": 6,
                "n_antibodies": 3,
                "reads_in_component": 14,
                "isotype_fraction": 0.5,
                "intracellular_fraction": 0.16666666666666666,
            },
            "3": {
                "n_umi": 3,
                "n_umi1": 2,
                "n_umi2": 1,
                "n_edges": 2,
                "n_antibodies": 2,
                "reads_in_component": 4,
                "isotype_fraction": 0.0,
                "intracellular_fraction": 0.6666666666666666,
            },
        },
        orient="index",
    )
    expected_obs.index.name = "component"
    expected_obs = expected_obs.astype(
        {
            "n_umi": np.uint64,
            "n_umi1": np.uint64,
            "n_umi2": np.uint64,
            "n_edges": np.uint64,
            "n_antibodies": np.uint32,
            "reads_in_component": np.uint64,
        }
    )
    assert_frame_equal(
        adata.obs.sort_index().sort_index(axis="columns"),
        expected_obs.sort_index(axis="columns"),
    )


def test_pna_edgelist_to_anndata_save_adata(pixelconnection, tmp_path):
    from pixelator.pna.config import pna_config

    panel = load_antibody_panel(pna_config, "proxiome-immuno-155")
    adata = pna_edgelist_to_anndata(pixelconnection, panel)

    adata.write_h5ad(tmp_path / "test.h5ad")
