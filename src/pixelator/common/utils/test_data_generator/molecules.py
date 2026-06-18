"""Marker and umi population for test data cell graphs.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pixelator.common.utils.test_data_generator.topology import generate_cell_graph

if TYPE_CHECKING:
    from pixelator.pna.config.panel import PNAAntibodyPanel


def generate_edgelist(
    n_cells: int,
    n_nodes: int,
    n_edges: int,
    min_neighbors: int,
    panel: PNAAntibodyPanel,
    n_crossing_edges: int = 1,
    rng=None,
) -> pl.DataFrame:
    """Generate a populated edge list for ``n_cells`` cells with crossing edges.

    Each cell is an independent graph (see :func:`generate_cell_graph`) populated
    with umis and markers (see :func:`populate_cell`). The per-cell edge lists are
    concatenated, and ``n_crossing_edges`` chimeric edges are added: for each, two
    random edges are sampled and a new edge joins the umi1/marker1 of the first
    with the umi2/marker2 of the second.

    Args:
        n_cells: number of cell graphs to generate.
        n_nodes: number of nodes per cell graph.
        n_edges: target number of edges per cell graph.
        min_neighbors: minimum number of neighbors per node.
        panel: antibody panel providing the available markers.
        n_crossing_edges: number of chimeric edges to add across cells.
        rng: a seed or numpy Generator for the random number generator.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker1``, ``umi2``, ``marker2``.
    """
    rng = np.random.default_rng(rng)

    # Generate and populate one edge list per cell, sharing the generator so no
    # two cells draw the same random sequence, then stack them.
    edgelist = pl.concat(
        populate_cell(
            generate_cell_graph(n_nodes, n_edges, min_neighbors, rng=rng),
            panel,
            rng=rng,
        )
        for _ in range(n_cells)
    )

    # Add crossing edges: join the umi1/marker1 of one random edge with the
    # umi2/marker2 of another random edge.
    first = edgelist[rng.integers(0, edgelist.height, size=n_crossing_edges)]
    second = edgelist[rng.integers(0, edgelist.height, size=n_crossing_edges)]
    crossing = pl.DataFrame(
        {
            "umi1": first["umi1"],
            "marker1": first["marker1"],
            "umi2": second["umi2"],
            "marker2": second["marker2"],
        }
    )
    return pl.concat([edgelist, crossing])


def populate_cell(
    edgelist: pl.DataFrame, panel: PNAAntibodyPanel, rng=None
) -> pl.DataFrame:
    """Populate a cell edge list with umis and markers.

    Args:
        edgelist: edge list with ``node1`` and ``node2`` node-index columns.
        panel: antibody panel providing the available markers.
        rng: a seed or numpy Generator for the random number generator.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker1``, ``umi2``, ``marker2``.
    """
    rng = np.random.default_rng(rng)
    node_umi_map = _assign_umis(edgelist, rng)
    node_umi_map = _assign_markers(node_umi_map, panel, rng)
    node_umi_map = _correlate_neighbors(node_umi_map, edgelist, rng)
    return (
        edgelist.join(
            node_umi_map.select(node1="node", umi1="umi", marker1="marker"), on="node1"
        )
        .join(
            node_umi_map.select(node2="node", umi2="umi", marker2="marker"), on="node2"
        )
        .select("umi1", "marker1", "umi2", "marker2")
    )


def _assign_umis(edgelist: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Map each node id to a random 56-bit umi."""
    nodes = pl.concat([edgelist["node1"], edgelist["node2"]]).unique().sort()
    return pl.DataFrame(
        {
            "node": nodes,
            "umi": rng.integers(0, 1 << 56, size=nodes.len(), dtype=np.int64),
        }
    )


def _marker_probabilities(
    panel: PNAAntibodyPanel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-marker sampling probabilities and the hashing-marker mask.

    Abundance tiers apply only to non-hashing markers: the first sixth of markers
    make 50% of all umis (high abundance), the next third make 40% (medium) and
    the last 50% make 10% (low). Hashing markers (``sample_hashing == "yes"``)
    are excluded from the tiers and sampled at the low-abundance per-marker rate.
    """
    df = panel.to_polars()
    markers = df["marker_id"].to_numpy()
    is_hashing = (df["sample_hashing"] == "yes").to_numpy()
    idx = np.flatnonzero(~is_hashing)

    n = idx.shape[0]
    n_high, n_medium = round(1.0 / 6.0 * n), round(2.0 / 6.0 * n)
    n_low = n - n_high - n_medium

    probs = np.full(markers.shape[0], 0.1 / n_low)
    probs[idx[:n_high]] = 0.5 / n_high
    probs[idx[n_high : n_high + n_medium]] = 0.4 / n_medium
    return markers, probs / probs.sum(), is_hashing


def _assign_markers(
    node_umi_map: pl.DataFrame, panel: PNAAntibodyPanel, rng: np.random.Generator
) -> pl.DataFrame:
    """Sample a marker per umi, then overwrite 5% with one random hashing marker."""
    markers, probs, is_hashing = _marker_probabilities(panel)
    hashing_marker = rng.choice(markers[is_hashing])
    return node_umi_map.with_columns(
        marker=rng.choice(markers, size=node_umi_map.height, p=probs)
    ).with_columns(
        marker=pl.when(pl.Series(rng.random(node_umi_map.height) < 0.05))
        .then(pl.lit(hashing_marker))
        .otherwise(pl.col("marker"))
    )


def _correlate_neighbors(
    node_umi_map: pl.DataFrame, edgelist: pl.DataFrame, rng: np.random.Generator
) -> pl.DataFrame:
    """Add spatial correlation: ~10% of nodes copy a neighbour's marker."""
    adj = pl.concat(
        [
            edgelist.select(node="node1", nbr="node2"),
            edgelist.select(node="node2", nbr="node1"),
        ]
    )
    return (
        adj.join(node_umi_map.select(nbr="node", nm="marker"), on="nbr")
        .group_by("node")
        .agg(pl.col("nm").first())
        .join(node_umi_map, on="node")
        .with_columns(
            marker=pl.when(pl.Series(rng.random(node_umi_map.height) < 0.10))
            .then(pl.col("nm"))
            .otherwise(pl.col("marker"))
        )
        .drop("nm")
        .sort("node")
    )
