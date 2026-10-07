"""Per-source patch bumps for concatenated panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections import defaultdict

from anndata import AnnData
from packaging.version import Version

from pixelator.common.utils import logger
from pixelator.pna.config.panel.antibody_panel import PanelSource, PNAAntibodyPanel
from pixelator.pna.config.panel.diff import PNAAntibodyPanelDiff
from pixelator.pna.config.panel.hashing import (
    collapsed_hashing_marker_id,
    split_hashing_marker_id,
)


def sample_calling_hashing_collapsed(
    hashing_ids: set[str],
    *,
    adata: AnnData | None = None,
    pxl_file_metadata: dict | None = None,
) -> bool:
    """Return whether sample calling has already collapsed hashing clones.

    Explicit ``hashing_collapsed`` on the pixel file metadata wins. Older
    files are inferred from ``original_hash_counts_*`` columns or from hashing
    clones that are absent from ``var``. Without an AnnData and without that
    key, there is nothing to infer from, so this returns False.
    """
    if pxl_file_metadata is not None and "hashing_collapsed" in pxl_file_metadata:
        return bool(pxl_file_metadata["hashing_collapsed"])
    if adata is None:
        return False
    if any(str(col).startswith("original_hash_counts_") for col in adata.obs.columns):
        return True
    if not hashing_ids:
        return False
    var_names = {str(name) for name in adata.var_names}
    return hashing_ids.isdisjoint(var_names)


def align_panel_patches(
    panels: list[PNAAntibodyPanel],
    adatas: list[AnnData] | None = None,
    *,
    pxl_file_metadata: list[dict] | None = None,
) -> tuple[list[PNAAntibodyPanel], list[dict[str, str]], list[dict[str, str]]]:
    """Bump each source to the newest patch carried by another panel.

    Sources match on name and product, and only when major and minor versions
    agree. A source that is not already on a panel is left alone.
    ``pxl_file_metadata`` is the pixel file metadata for each panel, in the same
    order, and carries ``hashing_collapsed`` when sample calling wrote the file.

    Returns:
        Updated panel copies, marker renames to apply to stored data (var,
        edgelist, proximity, layouts), and hashing-clone renames for
        ``original_hash_counts_*`` columns. Renames are keyed by input order.
    """
    updated = [panel.copy() for panel in panels]
    data_renames: list[dict[str, str]] = [{} for _ in panels]
    hash_renames: list[dict[str, str]] = [{} for _ in panels]

    families: dict[tuple[str, str, tuple[int, ...]], list[tuple[int, int, Version]]] = (
        defaultdict(list)
    )
    for panel_index, panel in enumerate(updated):
        for source_index, source in enumerate(panel.sources):
            identity = _source_identity(source)
            if identity is None:
                continue
            version = Version(source.metadata.version)
            minor = version.release[:2]
            families[(*identity, minor)].append((panel_index, source_index, version))

    for members in families.values():
        latest_idx, latest_source_idx, latest_version = max(
            members, key=lambda member: member[2]
        )
        latest_panel = updated[latest_idx].source_as_panel(latest_source_idx)
        for panel_index, source_index, version in members:
            if version == latest_version:
                continue
            current = updated[panel_index].source_as_panel(source_index)
            logger.info(
                "Upgrading panel source %s %s from %s to %s.",
                current.name,
                current.product,
                current.version,
                latest_panel.version,
            )
            diff = PNAAntibodyPanelDiff(current, latest_panel)
            clone_map = diff.changed_marker_ids()
            _validate_hashing_renames(current, latest_panel, clone_map)
            hashing_ids = current.hashing_marker_ids
            pxl_metadata = (
                None if pxl_file_metadata is None else pxl_file_metadata[panel_index]
            )
            collapsed = sample_calling_hashing_collapsed(
                hashing_ids,
                adata=None if adatas is None else adatas[panel_index],
                pxl_file_metadata=pxl_metadata,
            )
            if adatas is not None:
                _require_expected_markers(
                    adata=adatas[panel_index],
                    old_panel=current,
                    new_panel=latest_panel,
                    clone_map=clone_map,
                    collapsed=collapsed,
                )
            data_map = _data_rename_map(clone_map, hashing_ids, collapsed=collapsed)
            _merge_renames(data_renames[panel_index], data_map)
            _merge_renames(
                hash_renames[panel_index],
                {old: new for old, new in clone_map.items() if old in hashing_ids},
            )
            updated[panel_index] = updated[panel_index].replace_source(
                source_index, latest_panel
            )

    return updated, data_renames, hash_renames


def aligned_dataset_panel(panels: list[PNAAntibodyPanel]) -> PNAAntibodyPanel:
    """Return one panel after per-source patch alignment across files.

    Several files that describe the same sources collapse to a single panel.
    The order of ``--panel`` inputs does not have to match.
    """
    if len(panels) == 1:
        return panels[0]
    updated, _, _ = align_panel_patches(panels)
    first = updated[0]
    for other in updated[1:]:
        if first != other:
            raise ValueError(
                "Samples do not share the same panel sources after patch alignment."
            )
    return first


def _source_identity(source: PanelSource) -> tuple[str, str] | None:
    """Return ``(name, product)`` for a source, or None when product is unset."""
    product = source.metadata.product
    if not product:
        return None
    return (source.metadata.name, product)


def _data_rename_map(
    clone_map: dict[str, str], hashing_ids: set[str], *, collapsed: bool
) -> dict[str, str]:
    """Return marker renames, collapsing hashing clones when ``collapsed`` is set."""
    if not collapsed:
        return dict(clone_map)
    data_map = {old: new for old, new in clone_map.items() if old not in hashing_ids}
    families: dict[str, str] = {}
    for old, new in clone_map.items():
        if old not in hashing_ids:
            continue
        old_base = collapsed_hashing_marker_id(old)
        new_base = collapsed_hashing_marker_id(new)
        if old_base in families and families[old_base] != new_base:
            raise ValueError(
                f"Hashing markers with collapsed name {old_base!r} do not share "
                "a single new base name."
            )
        families[old_base] = new_base
    for old_base, new_base in families.items():
        if old_base != new_base:
            data_map[old_base] = new_base
    return data_map


def _validate_hashing_renames(
    old_panel: PNAAntibodyPanel,
    new_panel: PNAAntibodyPanel,
    clone_map: dict[str, str],
) -> None:
    """Reject a hashing rename that breaks the collapsed-name rules.

    Each of these fails:

    * ``B2M-1`` → ``C2M``: a hashing id must end with ``-<digits>``.
    * ``B2M-1`` → ``C2M-2``: the numeric suffix stays.
    * ``B2M-1`` → ``C2M-1`` and ``B2M-2`` → ``D2M-2``: clones in one family
      share one new base.
    * ``B2M`` stays while ``B2M-1`` → ``C2M-1``, or ``B2M`` → ``C2M`` while
      ``B2M-1`` stays: a non-hashing marker that already has the collapsed
      name renames with the family.
    * ``CD19`` → ``C2M`` while ``B2M-1`` → ``C2M-1``: the family must not
      land on a different non-hashing marker.
    """
    old_hashing = old_panel.hashing_marker_ids
    new_hashing = new_panel.hashing_marker_ids
    if not old_hashing:
        return

    families: dict[str, str] = {}
    for old in sorted(old_hashing):
        new = clone_map.get(old, old)
        if new not in new_hashing and old not in clone_map:
            continue
        old_parts = split_hashing_marker_id(old)
        new_parts = split_hashing_marker_id(new)
        if old_parts is None or new_parts is None:
            raise ValueError(
                "Hashing marker ids must end with -<digits> to be renamed "
                f"({old!r} -> {new!r})."
            )
        if old_parts[1] != new_parts[1]:
            raise ValueError(
                "Hashing marker rename may only change the base name, not the "
                f"hash group suffix {old_parts[1]}. Got {old!r} -> {new!r}."
            )
        old_base = collapsed_hashing_marker_id(old)
        new_base = collapsed_hashing_marker_id(new)
        if old_base in families and families[old_base] != new_base:
            raise ValueError(
                f"Hashing markers with collapsed name {old_base!r} must keep a "
                "single base name per hash group family."
            )
        families[old_base] = new_base

    non_hashing = {
        str(marker_id)
        for marker_id in old_panel.markers
        if str(marker_id) not in old_hashing
    }
    for old_base, new_base in families.items():
        if old_base in non_hashing:
            moved = clone_map.get(old_base, old_base)
            if moved != new_base:
                # TODO: we should think about if we want this check or not in the long run.
                # adding it here now we can always remove it later if we decide it's too strict.
                raise ValueError(
                    f"Hashing collapsed base {old_base!r} and non-hashing marker "
                    f"{old_base!r} must be renamed together "
                    f"(hashing maps to {new_base!r}, non-hashing maps to {moved!r})."
                )
        for other in non_hashing:
            if other == old_base:
                continue
            if clone_map.get(other, other) == new_base:
                raise ValueError(
                    f"Hashing collapsed base {old_base!r} -> {new_base!r} collides "
                    f"with non-hashing marker {other!r}."
                )


def _require_expected_markers(
    *,
    adata: AnnData,
    old_panel: PNAAntibodyPanel,
    new_panel: PNAAntibodyPanel,
    clone_map: dict[str, str],
    collapsed: bool,
) -> None:
    """Raise when a patch bump is missing a marker the file should still contain.

    After sample calling, hashing clones may be absent when the file is
    collapsed. A missing non-hashing marker still fails.
    """
    var_names = {str(name) for name in adata.var_names}
    old_hashing = old_panel.hashing_marker_ids
    new_hashing = new_panel.hashing_marker_ids
    old_markers = {str(marker_id) for marker_id in old_panel.markers}
    missing: list[str] = []
    for marker_id in old_panel.markers:
        marker = str(marker_id)
        if marker in var_names or marker in clone_map:
            continue
        if collapsed and marker in old_hashing:
            continue
        missing.append(marker)
    for marker_id in new_panel.markers:
        marker = str(marker_id)
        if marker in var_names or marker in old_markers:
            continue
        old_ids = [old for old, new in clone_map.items() if new == marker]
        if any(old in var_names for old in old_ids):
            continue
        if collapsed and (
            marker in new_hashing or any(old in old_hashing for old in old_ids)
        ):
            continue
        missing.append(marker)
    if missing:
        raise ValueError(
            "Row count mismatch in automatic panel patch version bump. "
            f"Missing markers: {sorted(set(missing))[:5]}"
        )


def _merge_renames(target: dict[str, str], extra: dict[str, str]) -> None:
    """Copy renames into ``target``, rejecting two destinations for one marker."""
    for old, new in extra.items():
        if old in target and target[old] != new:
            raise ValueError(
                f"Marker {old!r} is renamed to both {target[old]!r} and {new!r}."
            )
        if old != new:
            target[old] = new
