"""Apply marker renames from a panel patch bump onto stored data.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def stored_marker_ids(requested: set[str], old_to_new: dict[str, str]) -> set[str]:
    """Return the stored ids for marker names already renamed in memory.

    ``old_to_new`` maps a stored id to the id a patch bump exposes. A name
    with no entry is already the stored id.
    """
    new_to_old: dict[str, set[str]] = {}
    for old, new in old_to_new.items():
        if old == new:
            continue
        new_to_old.setdefault(new, set()).add(old)
    stored: set[str] = set()
    for name in requested:
        if name in new_to_old:
            stored.update(new_to_old[name])
        else:
            stored.add(name)
    return stored


def apply_marker_renames_to_frame(
    df: pd.DataFrame | object,
    renames_by_sample: dict[str, dict[str, str]],
    columns: Sequence[str],
):
    """Rename marker columns, using ``sample`` when maps differ across files."""
    if not isinstance(df, pl.DataFrame):
        raise TypeError("Expected a polars DataFrame.")
    active = {
        sample: mapping for sample, mapping in renames_by_sample.items() if mapping
    }
    if df.is_empty() or not active:
        return df

    rows = [
        {"__sample": sample, "__old": old, "__new": new}
        for sample, mapping in active.items()
        for old, new in mapping.items()
    ]
    map_df = pl.DataFrame(rows)
    has_sample = "sample" in df.columns
    for column in columns:
        if column not in df.columns:
            continue
        if has_sample:
            joined = df.join(
                map_df,
                left_on=["sample", column],
                right_on=["__sample", "__old"],
                how="left",
            )
            df = joined.with_columns(
                pl.coalesce([pl.col("__new"), pl.col(column)]).alias(column)
            ).drop("__new")
        else:
            only = next(iter(active.values()))
            df = df.with_columns(pl.col(column).replace(only).alias(column))
    return df


def apply_marker_renames_to_adata(adata: AnnData, mapping: dict[str, str]) -> None:
    """Rename ``var`` marker ids in place."""
    if not mapping:
        return
    new_names = [mapping.get(str(name), str(name)) for name in adata.var_names]
    if len(new_names) != len(set(new_names)):
        raise ValueError(
            "Panel patch bump would duplicate marker ids in var: "
            f"{sorted(name for name in new_names if new_names.count(name) > 1)}"
        )
    adata.var_names = new_names


def apply_hash_count_renames(adata: AnnData, mapping: dict[str, str]) -> None:
    """Rename ``original_hash_counts_*`` columns when hashing ids change."""
    if not mapping:
        return
    renames = {}
    for old, new in mapping.items():
        if old == new:
            continue
        old_col = f"original_hash_counts_{old}"
        new_col = f"original_hash_counts_{new}"
        if old_col in adata.obs.columns:
            if new_col in adata.obs.columns:
                raise ValueError(
                    f"Cannot rename hashing counts column {old_col} to {new_col}: "
                    "the target already exists."
                )
            renames[old_col] = new_col
    if renames:
        adata.obs.rename(columns=renames, inplace=True)
