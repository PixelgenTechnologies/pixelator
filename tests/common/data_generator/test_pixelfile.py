"""Tests for pxl file generation from populated edge lists.

Copyright © 2026 Pixelgen Technologies AB.
"""

import polars as pl
import pytest

from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.pixeldataset import read
from tests.common.data_generator import (
    generate_edgelist,
    write_pna_pxl,
)

N_CELLS = 3
N_EDGES = 1500
N_CROSSING_EDGES = 5
SAMPLE_NAME = "my_sample"


@pytest.fixture(name="real_panel", scope="module")
def real_panel_fixture():
    """A real panel; proxiome-v1 has no hashing markers (exercises that path)."""
    return load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")


@pytest.fixture(name="generated_edgelist", scope="module")
def generated_edgelist_fixture(real_panel):
    """A small multi-cell edge list with a few crossing edges."""
    return generate_edgelist(
        n_cells=N_CELLS,
        n_nodes=500,
        n_edges=N_EDGES,
        min_neighbors=20,
        panel=real_panel,
        n_crossing_edges=N_CROSSING_EDGES,
        rng=0,
    )


@pytest.fixture(name="written_pxl", scope="module")
def written_pxl_fixture(generated_edgelist, real_panel, tmp_path_factory):
    """Write the generated edge list to a pxl file once for the module."""
    path = tmp_path_factory.mktemp("pxl") / "synthetic.pxl"
    return write_pna_pxl(generated_edgelist, real_panel, path, sample_name=SAMPLE_NAME)


def test_write_pna_pxl_returns_existing_path(written_pxl):
    """The function returns the path it wrote to."""
    assert written_pxl.exists()


def test_write_pna_pxl_drops_crossing_edges(written_pxl, generated_edgelist):
    """Crossing edges (null component) are removed from the stored edge list."""
    edgelist = read(written_pxl).edgelist().to_polars()

    # every generated crossing edge is dropped
    assert edgelist.height == generated_edgelist.height - N_CROSSING_EDGES
    assert edgelist["component"].null_count() == 0


def test_write_pna_pxl_edgelist_schema(written_pxl):
    """The stored edge list is reshaped to the pxl schema."""
    edgelist = read(written_pxl).edgelist().to_polars()

    expected = {"umi1", "umi2", "marker_1", "marker_2", "component", "read_count"}
    assert expected <= set(edgelist.columns)
    # the random read counts from generate_edgelist are preserved
    assert edgelist["read_count"].min() >= 1
    assert edgelist["read_count"].n_unique() > 1


def test_write_pna_pxl_markers_from_panel(written_pxl, real_panel):
    """All markers in the stored edge list come from the panel."""
    edgelist = read(written_pxl).edgelist().to_polars()
    panel_markers = set(real_panel.markers)

    assert set(edgelist["marker_1"]) <= panel_markers
    assert set(edgelist["marker_2"]) <= panel_markers


def test_write_pna_pxl_components(written_pxl, generated_edgelist):
    """The AnnData has one observation per generated (non-crossing) cell."""
    adata = read(written_pxl).adata()

    expected_components = set(
        generated_edgelist.filter(pl.col("component").is_not_null())["component"]
    )
    assert adata.n_obs == N_CELLS
    assert set(adata.obs.index) == expected_components


def test_write_pna_pxl_adata_metrics(written_pxl):
    """The rebuilt AnnData excludes retired Tau metrics."""
    adata = read(written_pxl).adata()
    obs = adata.obs

    assert "isotype_fraction" in obs.columns
    assert "n_edges" in obs.columns


def test_write_pna_pxl_metadata(written_pxl, real_panel):
    """Metadata records the sample name, technology and panel identity."""
    metadata = read(written_pxl).metadata()

    assert set(metadata) == {SAMPLE_NAME}
    entry = metadata[SAMPLE_NAME]
    assert entry["technology"] == "single-cell-pna"
    assert entry["panel_name"] == real_panel.name
    assert entry["panel_version"] == real_panel.version
