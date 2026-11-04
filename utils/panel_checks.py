#!/usr/bin/env python3
"""Check the bundled PNA panels for version and sequence consistency.

Copyright © 2026 Pixelgen Technologies AB.

Two rules are enforced across the panel CSV files:

1. No two panels share both a product and a version.
2. Panels of one product that only differ in patch version (same major and
   minor) must have the same set of ``(sequence_1, sequence_2)`` pairs.
   Marker ids may change.

A panel without a product is identified by its name, as in ``align_panels``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from packaging.version import Version

from pixelator.common.config.panel import parse_panel_header_metadata

DEFAULT_PANEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pixelator"
    / "pna"
    / "resources"
    / "panels"
)
_SEQUENCE_COLUMNS = ["sequence_1", "sequence_2"]
_MAX_REPORTED_PAIRS = 5


@dataclass(frozen=True)
class PanelInfo:
    """The parts of a panel file that the checks need."""

    path: Path
    product: str
    version: Version
    sequences: frozenset[tuple[str, str]]


def load_panels(panel_dir: Path) -> list[PanelInfo]:
    """Read the header and sequence pairs of every panel CSV in a directory."""
    panels = []
    for path in sorted(panel_dir.glob("*.csv")):
        metadata = parse_panel_header_metadata(path)
        table = pd.read_csv(
            path, comment="#", usecols=_SEQUENCE_COLUMNS, dtype=str
        ).fillna("")
        panels.append(
            PanelInfo(
                path=path,
                product=metadata.product or metadata.name,
                version=Version(metadata.version),
                sequences=frozenset(
                    tuple(row)
                    for row in table[_SEQUENCE_COLUMNS].itertuples(index=False)
                ),
            )
        )
    return panels


def check_unique_versions(panels: Sequence[PanelInfo]) -> list[str]:
    """Return an error for each product and version found in several files."""
    seen: dict[tuple[str, Version], list[str]] = {}
    for panel in panels:
        seen.setdefault((panel.product, panel.version), []).append(panel.path.name)
    return [
        f"Product {product!r} has version {version} in several files: "
        + ", ".join(names)
        for (product, version), names in seen.items()
        if len(names) > 1
    ]


def _format_pairs(pairs: set[tuple[str, str]]) -> str:
    shown = sorted(pairs)[:_MAX_REPORTED_PAIRS]
    text = ", ".join(f"({first}, {second})" for first, second in shown)
    if len(pairs) > _MAX_REPORTED_PAIRS:
        text += f", and {len(pairs) - _MAX_REPORTED_PAIRS} more"
    return text


def check_patch_sequences(panels: Sequence[PanelInfo]) -> list[str]:
    """Return an error for each panel whose sequences differ from an older patch.

    Each panel is compared with the oldest panel of its product, major and
    minor version.
    """
    groups: dict[tuple[str, int, int], list[PanelInfo]] = {}
    for panel in panels:
        key = (panel.product, panel.version.major, panel.version.minor)
        groups.setdefault(key, []).append(panel)

    errors = []
    for group in groups.values():
        oldest, *others = sorted(group, key=lambda panel: panel.version)
        for panel in others:
            if panel.sequences == oldest.sequences:
                continue
            details = []
            added = set(panel.sequences - oldest.sequences)
            removed = set(oldest.sequences - panel.sequences)
            if added:
                details.append(f"added {_format_pairs(added)}")
            if removed:
                details.append(f"removed {_format_pairs(removed)}")
            errors.append(
                f"{panel.path.name} ({panel.version}) is a patch bump of "
                f"{oldest.path.name} ({oldest.version}) but its sequences "
                f"differ: {'; '.join(details)}"
            )
    return errors


def check_panels(panel_dir: Path) -> list[str]:
    """Run every check on the panels of a directory."""
    panels = load_panels(panel_dir)
    return check_unique_versions(panels) + check_patch_sequences(panels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the panels of the given directory, or of the bundled ones."""
    args = sys.argv[1:] if argv is None else argv
    panel_dir = Path(args[0]) if args else DEFAULT_PANEL_DIR
    errors = check_panels(panel_dir)
    if errors:
        print(f"Panel checks failed for {panel_dir}:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"All panels in {panel_dir} passed the checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
