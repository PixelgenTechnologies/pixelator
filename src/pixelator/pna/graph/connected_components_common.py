"""Shared connected-component utilities for PNA graph pipelines.

Kept under `pixelator.pna.graph` so the current graph step and older entry points
can reuse the same helpers without coupling the modern package to deprecated modules.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import xxhash

from pixelator.common.annotate.cell_calling import find_component_size_limits
from pixelator.common.exceptions import PixelatorBaseException

from .constants import MIN_PNA_COMPONENT_SIZE

logger = logging.getLogger(__name__)


class ConnectedComponentException(PixelatorBaseException):
    """Raised when connected-component computation or filtering fails."""


def hash_component(component: set[int]) -> str:
    """Hash a component deterministically based on its nodes.

    Note: this preserves the historical hashing behavior used by the legacy pipeline.
    """
    hasher = xxhash.xxh3_64()
    for node in sorted(component):
        hasher.update(int(node).to_bytes(length=8, byteorder="little"))
    return hasher.hexdigest()


def _name_components_with_umi_hashes(edgelist: pl.LazyFrame) -> pl.LazyFrame:
    comp_umis = (
        edgelist.group_by("component")
        .agg(pl.col("umi1").unique(), pl.col("umi2").unique())
        .collect()
    )
    comp_hashes: dict[object, str] = {}
    for comp, umi1, umi2 in comp_umis.rows():
        comp_hashes[comp] = hash_component(set(umi1 + umi2))

    return edgelist.with_columns(pl.col("component").replace_strict(comp_hashes))


def filter_components_by_size_dynamic(
    component_sizes: pl.DataFrame,
    lowest_passable_bound: int | None = MIN_PNA_COMPONENT_SIZE,
) -> tuple[pl.Series, int | None]:
    """Filter components by size using dynamic thresholds.

    :param component_sizes: DataFrame with columns `component` and `n_umi`.
    :returns: Components that pass the filter, and the computed lower bound.
    """
    if lowest_passable_bound is None:
        lowest_passable_bound = MIN_PNA_COMPONENT_SIZE

    lower_bound = find_component_size_limits(
        component_sizes=component_sizes["n_umi"].to_numpy(), direction="lower"
    )
    if lower_bound is None or lower_bound < lowest_passable_bound:
        lower_bound = lowest_passable_bound
        logger.warning(
            "Could not find a lower bound for component size filtering, will "
            "set the lower bound to " + str(lowest_passable_bound),
        )
    return (
        component_sizes.filter(pl.col("n_umi") >= lower_bound)["component"],
        lower_bound,
    )


def filter_components_by_size_hard_thresholds(
    component_sizes: pl.DataFrame,
    lower_bound: int | None,
    higher_bound: int | None,
) -> pl.Series:
    """Filter components by size using hard thresholds.

    :param component_sizes: DataFrame with columns `component` and `n_umi`.
    :param lower_bound: The lower bound for the component size.
    :param higher_bound: The higher bound for the component size.
    :returns: The `component` column for components that pass the filter.
    """
    if lower_bound is None:
        lower_bound = 0
    if higher_bound is None:
        higher_bound = np.iinfo(np.uint64).max
    return component_sizes.filter(
        (pl.col("n_umi") >= lower_bound) & (pl.col("n_umi") <= higher_bound)
    )["component"]
