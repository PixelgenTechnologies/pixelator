"""Align panels whose shared sources differ only in patch version.

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


@dataclass(frozen=True)
class PanelAlignment:
    """The result of aligning several panels.

    Attributes:
        rows: ``(sequence_1, sequence_2)`` to the newest row, as a dict with the
            ``marker_id`` and every panel column. Markers that only some of
            the panels contain, for example those of an add-on, are included.
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
            panel: A panel that was passed to
                :func:`~pixelator.pna.config.panel.align.align_panels`.

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
    """Align panels whose shared sources are different patches of one another.

    Panels may contain different sources. A base panel with add-on A and a base
    panel with add-on B align into base plus A plus B, with the base at its
    newest patch. Sources are matched on their product (their name when they
    have none).

    Markers are matched on their sequence pair, so a marker that was renamed in
    a newer patch is still recognised. Each pair maps to the row of the newest
    version that contains it.

    Args:
        panels: The panels to align, for example one per pxl file.

    Returns:
        The alignment of all the panels.

    Raises:
        ValueError: When no panel is given, or when sources of the same product
            have different major or minor versions.
    """
    if not panels:
        raise ValueError("At least one panel is required to align panels.")

    sources = [
        (
            source.metadata.product or source.metadata.name,
            Version(source.metadata.version),
            panel.source_as_panel(i),
        )
        for panel in panels
        for i, source in enumerate(panel.sources)
    ]

    source_versions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for key, version, _ in sources:
        source_versions[key].add((version.major, version.minor))
    conflicts = {key: v for key, v in source_versions.items() if len(v) > 1}
    if conflicts:
        found = "; ".join(
            f"{key}: {', '.join(f'{major}.{minor}' for major, minor in sorted(v))}"
            for key, v in conflicts.items()
        )
        raise ValueError(
            "Sources of the same product must have the same major and minor "
            f"version to be aligned. Found {found}"
        )

    # Oldest first, so that rows of a newer patch overwrite older ones.
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for _, _, source_panel in sorted(sources, key=lambda source: source[1]):
        for row in source_panel.df.reset_index().to_dict("records"):
            rows[(row["sequence_1"], row["sequence_2"])] = row
    return PanelAlignment(rows=rows)
