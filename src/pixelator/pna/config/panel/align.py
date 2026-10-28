"""Align panels whose shared sources differ only in patch version.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd
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

    _check_source_versions(sources)
    _check_marker_conflicts(sources, force)

    # Oldest first, so that rows of a newer patch overwrite older ones. The
    # versions of different products are not comparable, so products are
    # ordered as they first appear.
    product_order = {
        product: i for i, product in enumerate(dict.fromkeys(s[0] for s in sources))
    }
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


def _check_source_versions(
    sources: Sequence[tuple[str, Version, PNAAntibodyPanel]],
) -> None:
    """Raise if sources of one product differ in major or minor version."""
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


def _check_marker_conflicts(
    sources: Sequence[tuple[str, Version, PNAAntibodyPanel]], force: bool
) -> None:
    """Raise, or warn when forced, if the sources disagree on a marker."""
    entries = pd.concat(
        [
            panel.df[_SEQUENCE_COLUMNS]
            .reset_index()
            .assign(product=product, source=f"{product} {version}")
            for product, version, panel in sources
        ]
    )

    def where(group: pd.DataFrame) -> str:
        return ", ".join(sorted(group["source"].unique()))

    conflicts = [
        f"marker id {marker_id!r} has different sequences in {where(group)}"
        for marker_id, group in entries.groupby("marker_id")
        if len(group[_SEQUENCE_COLUMNS].drop_duplicates()) > 1
    ]
    # A marker may be renamed within a product, but not across products.
    conflicts += [
        f"sequences {pair} have different marker ids in {where(group)}"
        for pair, group in entries.groupby(_SEQUENCE_COLUMNS)
        if group["marker_id"].nunique() > 1 and group["product"].nunique() > 1
    ]
    if not conflicts:
        return

    message = "\n".join(
        ["Panels have conflicting markers:"]
        + [f"  {c}" for c in conflicts[:_MAX_REPORTED_CONFLICTS]]
    )
    if len(conflicts) > _MAX_REPORTED_CONFLICTS:
        message += f"\n  ... and {len(conflicts) - _MAX_REPORTED_CONFLICTS} more"
    if not force:
        raise PanelConflictError(f"{message}\nPass force=True to load them anyway.")
    logger.warning("%s\nForced to continue, matching markers on sequences.", message)
