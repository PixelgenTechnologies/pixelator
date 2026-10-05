"""Apply a panel patch bump to AnnData marker ids.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


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
