"""Marker and umi population for test data cell graphs.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from tests.common.data_generator.topology import generate_cell_graph

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
    *,
    hashing_indices: Sequence[int] | None = None,
    n_cell_types: int = 1,
    cell_type_effect: float = 1.0,
    n_shared_markers: int = 0,
) -> pl.DataFrame:
    """Generate a populated edge list for ``n_cells`` cells with crossing edges.

    Each cell is an independent graph (see :func:`generate_cell_graph`) populated
    with umis and markers (see :func:`populate_cell`). The per-cell edge lists are
    concatenated and tagged with a unique ``component`` id, and
    ``n_crossing_edges`` chimeric edges are added: for each, two random edges are
    sampled and a new edge joins the umi1/marker_1 of the first with the
    umi2/marker_2 of the second. Crossing edges have a null ``component``.

    Each cell also gets a cell type, cycled within each hashing index so that
    every hashing index carries every cell type (see :func:`_cell_types_per_cell`).

    Args:
        n_cells: number of cell graphs to generate.
        n_nodes: number of nodes per cell graph.
        n_edges: target number of edges per cell graph.
        min_neighbors: minimum number of candidate neighbors per node, before filtering.
        panel: antibody panel providing the available markers.
        n_crossing_edges: number of chimeric edges to add across cells.
        hashing_fraction: fraction of each cell's umis assigned to hashing markers.
        rng: a seed or numpy Generator for the random number generator.
        hashing_indices: hashing indices to assign to cells. ``None`` uses every
            index in the panel.
        n_cell_types: number of cell types; each boosts its own block of
            high-abundance markers (see :func:`_marker_probabilities`).
        cell_type_effect: factor applied to the probabilities of a cell type's
            marker block. ``1.0`` plants no cell-type signal.
        n_shared_markers: number of high-abundance markers that receive the
            ``cell_type_effect`` boost in every cell, whatever its cell type.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker_1``, ``umi2``,
        ``marker_2``, ``component`` (null for crossing edges) and a random
        positive ``read_count`` per edge.
    """
    rng = np.random.default_rng(rng)

    # Assign one hashing index per cell so that every index is used at least
    # once (when there are enough cells).
    cell_hashing_indices = _hashing_indices_per_cell(
        n_cells, panel, rng, hashing_indices
    )
    cell_types = _cell_types_per_cell(cell_hashing_indices, n_cell_types)

    # Generate and populate one edge list per cell, sharing the generator so no
    # two cells draw the same random sequence, then stack them. Each cell is
    # tagged with a unique component id.
    edgelist = pl.concat(
        populate_cell(
            generate_cell_graph(n_nodes, n_edges, min_neighbors, rng=rng),
            panel,
            None if hashing_index is None else int(hashing_index),
            hashing_fraction,
            rng=rng,
            cell_type=int(cell_type),
            n_cell_types=n_cell_types,
            cell_type_effect=cell_type_effect,
            n_shared_markers=n_shared_markers,
        ).with_columns(component=pl.lit(f"{cell:016x}"))
        for cell, (hashing_index, cell_type) in enumerate(
            zip(cell_hashing_indices, cell_types)
        )
    )

    # Add crossing edges: join the umi1/marker_1 of one random edge with the
    # umi2/marker_2 of another random edge. These belong to no single cell, so
    # their component is null.
    first = edgelist[rng.integers(0, edgelist.height, size=n_crossing_edges)]
    second = edgelist[rng.integers(0, edgelist.height, size=n_crossing_edges)]
    crossing = pl.DataFrame(
        {
            "umi1": first["umi1"],
            "marker_1": first["marker_1"],
            "umi2": second["umi2"],
            "marker_2": second["marker_2"],
            "component": pl.Series([None] * n_crossing_edges, dtype=pl.Utf8),
        }
    )

    # Assign a random positive read count to every edge.
    populated = pl.concat([edgelist, crossing])
    return populated.with_columns(
        read_count=pl.Series(
            rng.integers(1, 101, size=populated.height), dtype=pl.UInt32
        )
    )


def populate_cell(
    edgelist: pl.DataFrame,
    panel: PNAAntibodyPanel,
    hashing_index: int | None = None,
    hashing_fraction: float = 0.2,
    rng=None,
    *,
    cell_type: int = 0,
    n_cell_types: int = 1,
    cell_type_effect: float = 1.0,
    n_shared_markers: int = 0,
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
        cell_type: cell type of this cell, in ``[0, n_cell_types)``.
        n_cell_types: number of cell types the marker blocks are split into.
        cell_type_effect: factor applied to the probabilities of the cell
            type's marker block.
        n_shared_markers: number of high-abundance markers boosted by
            ``cell_type_effect`` in every cell.

    Returns:
        A polars DataFrame with columns ``umi1``, ``marker_1``, ``umi2``, ``marker_2``.
    """
    rng = np.random.default_rng(rng)
    node_umi_map = _assign_umis(edgelist, rng)
    node_umi_map = _assign_markers(
        node_umi_map,
        panel,
        hashing_index,
        hashing_fraction,
        rng,
        cell_type=cell_type,
        n_cell_types=n_cell_types,
        cell_type_effect=cell_type_effect,
        n_shared_markers=n_shared_markers,
    )
    node_umi_map = _correlate_neighbors(node_umi_map, edgelist, rng)
    return (
        edgelist.join(
            node_umi_map.select(node1="node", umi1="umi", marker_1="marker"), on="node1"
        )
        .join(
            node_umi_map.select(node2="node", umi2="umi", marker_2="marker"), on="node2"
        )
        .select("umi1", "marker_1", "umi2", "marker_2")
    )


def _hashing_indices_per_cell(
    n_cells: int,
    panel: PNAAntibodyPanel,
    rng: np.random.Generator,
    hashing_indices: Sequence[int] | None = None,
) -> np.ndarray | list[None]:
    """Assign a hashing index to each cell, covering every requested index.

    The requested hashing indices (by default every ``-X`` suffix of the
    panel's hashing markers) are tiled to ``n_cells`` so each is used at least
    once when ``n_cells`` is at least the number of indices, then shuffled
    across cells. When the panel has no hashing markers, ``[None] * n_cells`` is
    returned so each cell skips the hashing overwrite.
    """
    df = panel.to_polars()
    hashing = df["marker_id"].to_numpy()[_hashing_mask(df)]
    if hashing.size == 0:
        return [None] * n_cells
    if hashing_indices is None:
        indices = np.unique([int(m.rsplit("-", 1)[-1]) for m in hashing])
    else:
        indices = np.unique(hashing_indices)
    cell_indices = np.resize(indices, n_cells)
    rng.shuffle(cell_indices)
    return cell_indices


def _cell_types_per_cell(
    cell_hashing_indices: np.ndarray | list[None], n_cell_types: int
) -> np.ndarray:
    """Cycle cell types within each hashing index.

    The k-th cell of a given hashing index gets cell type ``k % n_cell_types``,
    so every index carries every cell type once it has at least ``n_cell_types``
    cells, whatever the seed. The shuffled hashing assignment decides which cell
    gets which type. Cells without a hashing index are cycled as one group.
    """
    keys = np.array([-1 if h is None else h for h in cell_hashing_indices])
    cell_types = np.empty(keys.size, dtype=int)
    for key in np.unique(keys):
        where = np.flatnonzero(keys == key)
        cell_types[where] = np.arange(where.size) % n_cell_types
    return cell_types


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
    cell_type: int = 0,
    n_cell_types: int = 1,
    cell_type_effect: float = 1.0,
    n_shared_markers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-marker sampling probabilities and the hashing-marker mask.

    Abundance tiers apply only to non-hashing markers: the first sixth make 50%
    of all umis (high abundance) and the next third make 40% (medium). The
    low-abundance tier (the remaining non-hashing markers) and the hashing
    markers (``sample_hashing == "yes"``) together share the final 10%, every one
    of them sampled at the same per-marker rate ``0.1 / (n_low + n_hashing)``.
    Folding the hashing markers into the low tier's budget keeps the high and
    medium tiers at 50% and 40% regardless of how many hashing markers the panel
    has. The result is normalized to sum to one (it already does up to
    floating-point error).

    The first ``n_shared_markers`` non-control markers of the high tier are
    shared by every cell type; the remaining ones are split into
    ``n_cell_types`` contiguous blocks, in panel order. The shared markers and
    the block of ``cell_type`` are multiplied by ``cell_type_effect`` before
    normalization. The markers depend only on the panel, so a cell type boosts
    the same markers in every generation call.

    NB: for the hashing markers, these probabilities correspond to the noise
    background. True hashing markers are applied separately.
    """
    df = panel.to_polars()
    markers = df["marker_id"].to_numpy()
    is_hashing = _hashing_mask(df)
    idx = np.flatnonzero(~is_hashing)

    n = idx.shape[0]
    n_high, n_medium = round(1.0 / 6.0 * n), round(2.0 / 6.0 * n)
    n_low = n - n_high - n_medium

    probs = np.full(markers.shape[0], 0.1 / (n_low + is_hashing.sum()))
    probs[idx[:n_high]] = 0.5 / n_high
    probs[idx[n_high : n_high + n_medium]] = 0.4 / n_medium
    is_control = df["control"].to_numpy()
    high = idx[:n_high][~is_control[idx[:n_high]]]
    blocks = np.array_split(high[n_shared_markers:], n_cell_types)
    probs[high[:n_shared_markers]] *= cell_type_effect
    probs[blocks[cell_type]] *= cell_type_effect
    return markers, probs / probs.sum(), is_hashing


def _assign_markers(
    node_umi_map: pl.DataFrame,
    panel: PNAAntibodyPanel,
    hashing_index: int | None,
    hashing_fraction: float,
    rng: np.random.Generator,
    cell_type: int = 0,
    n_cell_types: int = 1,
    cell_type_effect: float = 1.0,
    n_shared_markers: int = 0,
) -> pl.DataFrame:
    """Sample a marker per umi, then overwrite a fraction with hashing markers.

    A ``hashing_fraction`` of the umis is selected for hashing and assigned
    uniformly among all hashing markers sharing ``hashing_index`` (the ``-X``
    suffix in the name). When ``hashing_index`` is ``None`` or the panel has no
    matching hashing markers, the overwrite is skipped entirely.
    """
    markers, probs, is_hashing = _marker_probabilities(
        panel, cell_type, n_cell_types, cell_type_effect, n_shared_markers
    )
    n = node_umi_map.height
    node_umi_map = node_umi_map.with_columns(
        marker=rng.choice(markers, size=n, p=probs)
    )

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
