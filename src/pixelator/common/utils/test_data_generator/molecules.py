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
    hashing_fraction: float = 0.2,
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
        hashing_fraction: fraction of each cell's umis assigned to hashing markers.
        rng: a seed or numpy Generator for the random number generator.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker1``, ``umi2``, ``marker2``.
    """
    rng = np.random.default_rng(rng)

    # Assign one hashing index per cell so that every index in the panel is used
    # at least once (when there are enough cells).
    cell_hashing_indices = _hashing_indices_per_cell(n_cells, panel, rng)

    # Generate and populate one edge list per cell, sharing the generator so no
    # two cells draw the same random sequence, then stack them.
    edgelist = pl.concat(
        populate_cell(
            generate_cell_graph(n_nodes, n_edges, min_neighbors, rng=rng),
            panel,
            None if hashing_index is None else int(hashing_index),
            hashing_fraction,
            rng=rng,
        )
        for hashing_index in cell_hashing_indices
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
    edgelist: pl.DataFrame,
    panel: PNAAntibodyPanel,
    hashing_index: int | None = None,
    hashing_fraction: float = 0.2,
    rng=None,
) -> pl.DataFrame:
    """Populate a cell edge list with umis and markers.

    Args:
        edgelist: edge list with ``node1`` and ``node2`` node-index columns.
        panel: antibody panel providing the available markers.
        hashing_index: hashing index (the ``-X`` suffix) whose markers receive
            the hashing overwrite for this cell. ``None`` (or a panel without
            hashing markers) skips the hashing overwrite.
        hashing_fraction: fraction of the cell's umis assigned to the hashing
            markers of ``hashing_index``.
        rng: a seed or numpy Generator for the random number generator.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker1``, ``umi2``, ``marker2``.
    """
    rng = np.random.default_rng(rng)
    node_umi_map = _assign_umis(edgelist, rng)
    node_umi_map = _assign_markers(
        node_umi_map, panel, hashing_index, hashing_fraction, rng
    )
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


def _hashing_indices_per_cell(
    n_cells: int, panel: PNAAntibodyPanel, rng: np.random.Generator
) -> np.ndarray | list[None]:
    """Assign a hashing index to each cell, covering every panel index.

    The unique hashing indices (the ``-X`` suffix of the hashing markers) are
    tiled to ``n_cells`` so each is used at least once when ``n_cells`` is at
    least the number of indices, then shuffled across cells. When the panel has
    no hashing markers, ``[None] * n_cells`` is returned so each cell skips the
    hashing overwrite.
    """
    df = panel.to_polars()
    hashing = df["marker_id"].to_numpy()[_hashing_mask(df)]
    if hashing.size == 0:
        return [None] * n_cells
    indices = np.unique([int(m.rsplit("-", 1)[-1]) for m in hashing])
    cell_indices = np.resize(indices, n_cells)
    rng.shuffle(cell_indices)
    return cell_indices


def _assign_umis(edgelist: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Map each node id to a random 56-bit umi."""
    nodes = pl.concat([edgelist["node1"], edgelist["node2"]]).unique().sort()
    return pl.DataFrame(
        {
            "node": nodes,
            "umi": rng.integers(0, 1 << 56, size=nodes.len(), dtype=np.int64),
        }
    )


def _hashing_mask(df: pl.DataFrame) -> np.ndarray:
    """Boolean mask of hashing markers; all ``False`` when the panel has none.

    Panels without a ``sample_hashing`` column (or with no ``"yes"`` entries) are
    treated as having no hashing markers, so hashing degrades gracefully.
    """
    if "sample_hashing" in df.columns:
        return (df["sample_hashing"] == "yes").to_numpy()
    return np.zeros(df.height, dtype=bool)


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
    is_hashing = _hashing_mask(df)
    idx = np.flatnonzero(~is_hashing)

    n = idx.shape[0]
    n_high, n_medium = round(1.0 / 6.0 * n), round(2.0 / 6.0 * n)
    n_low = n - n_high - n_medium

    probs = np.full(markers.shape[0], 0.1 / n_low)
    probs[idx[:n_high]] = 0.5 / n_high
    probs[idx[n_high : n_high + n_medium]] = 0.4 / n_medium
    return markers, probs / probs.sum(), is_hashing


def _assign_markers(
    node_umi_map: pl.DataFrame,
    panel: PNAAntibodyPanel,
    hashing_index: int | None,
    hashing_fraction: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Sample a marker per umi, then overwrite a fraction with hashing markers.

    A ``hashing_fraction`` of the umis is selected for hashing and assigned
    uniformly among all hashing markers sharing ``hashing_index`` (the ``-X``
    suffix in the name). When ``hashing_index`` is ``None`` or the panel has no
    matching hashing markers, the overwrite is skipped entirely.
    """
    markers, probs, is_hashing = _marker_probabilities(panel)
    n = node_umi_map.height
    node_umi_map = node_umi_map.with_columns(marker=rng.choice(markers, size=n, p=probs))

    if hashing_index is None:
        return node_umi_map

    hashing_markers = markers[is_hashing]
    index = np.array([int(m.rsplit("-", 1)[-1]) for m in hashing_markers])
    chosen = hashing_markers[index == hashing_index]
    if chosen.size == 0:
        return node_umi_map

    return node_umi_map.with_columns(
        marker=pl.when(pl.Series(rng.random(n) < hashing_fraction))
        .then(pl.Series(rng.choice(chosen, size=n)))
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
