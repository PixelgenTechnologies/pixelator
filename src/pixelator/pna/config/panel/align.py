"""Align panels that differ only in their patch version.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from packaging.version import Version

from pixelator.pna.config.panel.antibody_panel import PanelSource, PNAAntibodyPanel
from pixelator.pna.config.panel.hashing import collapsed_hashing_marker_id

_SEQUENCE_COLUMNS = ["sequence_1", "sequence_2"]

# (name, product, major, minor): what a patch version must leave unchanged.
_Family = tuple[str, str | None, int, int]


def _source_family(source: PanelSource) -> _Family:
    """Return the family of a panel source, which excludes the patch version."""
    version = Version(source.metadata.version)
    return (source.metadata.name, source.metadata.product, version.major, version.minor)


def _panel_families(panel: PNAAntibodyPanel) -> frozenset[_Family]:
    return frozenset(_source_family(source) for source in panel.sources)


def _format_family(family: _Family) -> str:
    name, product, major, minor = family
    product_part = f" (product {product})" if product else ""
    return f"{name}{product_part} {major}.{minor}.x"


@dataclass(frozen=True)
class PanelAlignment:
    """The newest known row for every marker, keyed by its sequence pair.

    Attributes:
        rows: ``(sequence_1, sequence_2)`` to the newest row, as a dict with the
            ``marker_id`` and every panel column.
    """

    rows: dict[tuple[str, str], dict[str, Any]]

    def marker_renames(self, panel: PNAAntibodyPanel) -> dict[str, str]:
        """Return ``old marker id -> new marker id`` for one input panel.

        Only ids that change are included. Hashing markers get two kinds of
        entries, because sample calling collapses their clone ids in the data:

        - the clone id, ``B2M-1 -> B3M-1``
        - the collapsed id, ``B2M -> B3M``

        The collapsed entry is left out when the clones of one base do not
        agree on the new base, or when the base id is itself a marker of the
        panel, since the two could not be told apart in collapsed data.

        Args:
            panel: A panel that was passed to :func:`align_panels`.

        Returns:
            The renames, empty when the panel is already up to date.
        """
        new_ids: dict[str, str] = {}
        for marker_id, sequences in zip(
            panel.df.index, panel.df[_SEQUENCE_COLUMNS].itertuples(index=False)
        ):
            row = self.rows.get(tuple(sequences))
            new_ids[str(marker_id)] = str(row["marker_id"]) if row else str(marker_id)
        renames = {old: new for old, new in new_ids.items() if old != new}

        # new collapsed ids reached from each old collapsed id
        new_bases: dict[str, set[str]] = defaultdict(set)
        for old in panel.hashing_marker_ids:
            new_bases[collapsed_hashing_marker_id(old)].add(
                collapsed_hashing_marker_id(new_ids[old])
            )
        for old_base, candidates in new_bases.items():
            if len(candidates) != 1 or old_base in new_ids:
                continue
            (new_base,) = candidates
            if new_base != old_base:
                renames[old_base] = new_base
        return renames


def align_panels(panels: Sequence[PNAAntibodyPanel]) -> PanelAlignment:
    """Align panels that are different patches of the same panel.

    Markers are matched on their sequence pair, so a marker that was renamed in
    a newer patch is still recognised. Each pair maps to the row of the newest
    patch that contains it, including markers only an older patch has.

    Args:
        panels: The panels to align, for example one per pxl file.

    Returns:
        The alignment of all the panels.

    Raises:
        ValueError: When no panel is given, or when the panels do not all have
            the same sources up to their patch version, meaning the same name,
            product, major and minor version.
    """
    if not panels:
        raise ValueError("At least one panel is required to align panels.")

    families = [_panel_families(panel) for panel in panels]
    if any(family != families[0] for family in families):
        found = sorted({_format_family(f) for family in families for f in family})
        raise ValueError(
            "Panels can only be aligned when they differ in patch version only. "
            f"Found: {', '.join(found)}"
        )

    versioned_sources = [
        (Version(source.metadata.version), panel.source_as_panel(index))
        for panel in panels
        for index, source in enumerate(panel.sources)
    ]
    # Oldest first, so that rows of a newer patch overwrite older ones.
    versioned_sources.sort(key=lambda versioned: versioned[0])

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for _, source_panel in versioned_sources:
        for row in source_panel.df.reset_index().to_dict("records"):
            rows[(row["sequence_1"], row["sequence_2"])] = row
    return PanelAlignment(rows=rows)
