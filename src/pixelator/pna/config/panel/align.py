"""Align panels whose shared sources differ only in patch version.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from packaging.version import Version

from pixelator.common.utils import logger
from pixelator.pna.config.panel.antibody_panel import PanelSource, PNAAntibodyPanel
from pixelator.pna.config.panel.hashing import collapsed_hashing_marker_id

_SEQUENCE_COLUMNS = ["sequence_1", "sequence_2"]
_MAX_REPORTED_CONFLICTS = 10


class PanelConflictError(ValueError):
    """Raised when panels disagree on which marker a sequence pair or id is."""


@dataclass(frozen=True)
class PanelAlignment:
    """The result of aligning several panels.

    Attributes:
        rows: ``(sequence_1, sequence_2)`` to the newest row, as a dict with the
            ``marker_id`` and every panel column. Markers that only an older
            version contains are included.
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


def align_panels(
    panels: Sequence[PNAAntibodyPanel], force: bool = False
) -> PanelAlignment:
    """Align panels whose shared sources are different patches of one another.

    Panels may contain different sources. A base panel with add-on A and a base
    panel with add-on B align into base plus A plus B, with the base at its
    newest patch. Sources are matched on their product (their name when they
    have none).

    Markers are matched on their sequence pair, so a marker that was renamed in
    a newer patch is still recognised. Each pair maps to the row of the newest
    version that contains it. Versions are only compared within a product.
    Sources of different products, and sources of one product with the same
    version, are ordered as they come in ``panels``, and the last one wins.

    Two kinds of conflict are rejected, because merging them would silently mix
    different markers:

    - one marker id with more than one sequence pair
    - one sequence pair with different marker ids in sources of different
      products

    Args:
        panels: The panels to align, for example one per pxl file.
        force: Log the conflicts as a warning instead of raising. Conflicting
            markers are then matched on their sequence pair.

    Returns:
        The alignment of all the panels.

    Raises:
        PanelConflictError: When the panels have conflicting markers and
            ``force`` is not set.
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

    marker_conflicts = _find_marker_conflicts(sources)
    if marker_conflicts:
        shown = marker_conflicts[:_MAX_REPORTED_CONFLICTS]
        more = len(marker_conflicts) - len(shown)
        message = "Panels have conflicting markers:\n" + "\n".join(
            f"  {conflict}" for conflict in shown
        )
        if more > 0:
            message += f"\n  ... and {more} more"
        if not force:
            raise PanelConflictError(f"{message}\nPass force=True to load them anyway.")
        logger.warning(
            "%s\nForced to continue, matching markers on sequences.", message
        )

    # Oldest first, so that rows of a newer patch overwrite older ones. The
    # versions of different products are not comparable, so products are
    # ordered as they first appear.
    product_order: dict[str, int] = {}
    for key, _, _ in sources:
        product_order.setdefault(key, len(product_order))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for _, _, source_panel in sorted(
        sources, key=lambda source: (product_order[source[0]], source[1])
    ):
        for row in source_panel.df.reset_index().to_dict("records"):
            pair = (row["sequence_1"], row["sequence_2"])
            # Re-insert so that the order of the rows follows their recency.
            rows.pop(pair, None)
            rows[pair] = row
    return PanelAlignment(rows=rows)


def _find_marker_conflicts(
    sources: Sequence[tuple[str, Version, PNAAntibodyPanel]],
) -> list[str]:
    """Describe the marker ids and sequence pairs the sources disagree on."""
    # marker id -> sequence pair -> sources, and the reverse with marker ids
    pairs_of_id: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    ids_of_pair: dict[tuple[str, str], dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for product, version, source_panel in sources:
        label = f"{product} {version}"
        for marker_id, sequences in zip(
            source_panel.df.index,
            source_panel.df[_SEQUENCE_COLUMNS].itertuples(index=False),
        ):
            pair = (str(sequences[0]), str(sequences[1]))
            pairs_of_id[str(marker_id)][pair].add(label)
            # Keyed on the product too, to tell renames within a product apart.
            ids_of_pair[pair][(str(marker_id), product)].add(label)

    conflicts = []
    for marker_id, pairs in pairs_of_id.items():
        if len(pairs) > 1:
            where = ", ".join(
                f"{'/'.join(sorted(labels))} ({pair[0]}, {pair[1]})"
                for pair, labels in pairs.items()
            )
            conflicts.append(
                f"marker id {marker_id!r} has different sequences: {where}"
            )
    for pair, entries in ids_of_pair.items():
        # Renamed ids are legal within a product, not across products.
        if any(
            id_a != id_b and product_a != product_b
            for (id_a, product_a), (id_b, product_b) in combinations(entries, 2)
        ):
            where = ", ".join(
                f"{'/'.join(sorted(labels))} as {marker_id!r}"
                for (marker_id, _), labels in entries.items()
            )
            conflicts.append(
                f"sequences ({pair[0]}, {pair[1]}) have different marker ids: {where}"
            )
    return conflicts
