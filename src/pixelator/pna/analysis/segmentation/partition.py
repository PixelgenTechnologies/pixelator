"""Protein counts aggregated by node partition.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import pandas as pd

from pixelator.pna.graph import PNAGraph


def partition_counts(
    graph: PNAGraph,
    partition: Sequence[Any] | pd.Series | None = None,
    partition_column: str | None = None,
) -> pd.DataFrame:
    """Sum node protein counts by partition group.

    Each node belongs to one group (for example ``cell1``, ``cell2``,
    ``interface``, or ``other`` after conjugate segmentation). This returns
    the protein count matrix collapsed to those groups: one row per partition
    and one column per protein.

    Provide exactly one of ``partition`` or ``partition_column``. A positional
    ``partition`` vector is aligned to graph node order. A :class:`~pandas.Series`
    whose index matches the node names is aligned by name. A pandas
    :class:`~pandas.Categorical` keeps its category order, including unused
    levels as all-zero rows.

    Args:
        graph: A :class:`~pixelator.pna.graph.PNAGraph` with node marker
            counts, typically a single component from
            ``dataset.edgelist().iterator()``.
        partition: Labels for every node. Either this or
            ``partition_column`` must be provided.
        partition_column: Name of a node attribute that holds the partition
            labels.

    Returns:
        A DataFrame with one row per partition group and one column per
        protein. Values are the summed node marker counts in that group.

    Raises:
        TypeError: If ``graph`` is not a :class:`~pixelator.pna.graph.PNAGraph`.
        ValueError: If neither or both of ``partition`` and
            ``partition_column`` are given, if ``partition`` has the wrong
            length, or if ``partition_column`` is missing from the graph.

    Examples:
        Sum markers by a vector of node labels, or by a node attribute::

            from pixelator.pna.analysis import partition_counts

            counts = partition_counts(graph, partition=labels)
            counts = partition_counts(graph, partition_column="compartment")

    See Also:
        ``partition_counts`` in pixelatorR, the equivalent function for
        R users.

    """
    if not isinstance(graph, PNAGraph):
        raise TypeError("graph must be a PNAGraph.")
    if partition is None and partition_column is None:
        raise ValueError("Either `partition` or `partition_column` must be provided.")
    if partition is not None and partition_column is not None:
        raise ValueError(
            "One of `partition` or `partition_column` must be provided, not both."
        )

    node_order = list(graph.raw.nodes())
    counts = graph.node_marker_counts.reindex(node_order)
    if partition_column is not None:
        labels = _labels_from_column(graph, partition_column, node_order)
    else:
        labels = _align_partition(partition, node_order)

    grouped = counts.groupby(labels, sort=False, observed=False, dropna=False).sum()
    grouped = grouped.reindex(_group_levels(labels), fill_value=0)
    grouped.index.name = "partition"
    return grouped


def _labels_from_column(
    graph: PNAGraph, partition_column: str, node_order: list[Any]
) -> pd.Series:
    if partition_column not in graph.vs.attributes():
        raise ValueError(
            f"Column '{partition_column}' not found in cell graph node attributes."
        )
    attrs = nx.get_node_attributes(graph.raw, partition_column)
    missing = [node for node in node_order if node not in attrs]
    if missing:
        raise ValueError(f"Column '{partition_column}' is missing on some graph nodes.")
    return pd.Series([attrs[node] for node in node_order], index=node_order)


def _align_partition(
    partition: Sequence[Any] | pd.Series, node_order: list[Any]
) -> pd.Series:
    n_nodes = len(node_order)
    if isinstance(partition, pd.Series) and set(partition.index) == set(node_order):
        return partition.reindex(node_order)

    if isinstance(partition, pd.Series):
        values = partition.tolist()
        categories = (
            partition.cat.categories
            if isinstance(partition.dtype, pd.CategoricalDtype)
            else None
        )
    elif isinstance(partition, pd.Categorical):
        values = partition.tolist()
        categories = partition.categories
    else:
        values = list(partition)
        categories = None

    if len(values) != n_nodes:
        raise ValueError(
            "Length of `partition` must match the number of nodes in the cell graph."
        )
    if categories is not None:
        return pd.Series(
            pd.Categorical(values, categories=categories), index=node_order
        )
    return pd.Series(values, index=node_order)


def _group_levels(labels: pd.Series) -> pd.Index:
    if isinstance(labels.dtype, pd.CategoricalDtype):
        return pd.Index(labels.cat.categories)
    return pd.Index(pd.unique(labels))
