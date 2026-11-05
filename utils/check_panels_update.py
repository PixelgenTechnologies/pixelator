#!/usr/bin/env python3
"""Check that the bundled PNA panels modified since a base ref were updated validly.

Copyright © 2026 Pixelgen Technologies AB.

A patch bump edits a panel file in place. Every panel file modified since the
merge-base of ``--base`` and HEAD must satisfy:

1. The major and minor version are unchanged and the version increased.
2. The set of ``(sequence_1, sequence_2)`` pairs is unchanged. Marker ids may
   change.

Added, deleted and renamed files have no earlier version to compare with and
are skipped.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from panel_files import DEFAULT_PANEL_DIR, PanelInfo, load_panel

_MAX_REPORTED_PAIRS = 5


def _format_pairs(pairs: set[tuple[str, str]]) -> str:
    shown = sorted(pairs)[:_MAX_REPORTED_PAIRS]
    text = ", ".join(f"({first}, {second})" for first, second in shown)
    if len(pairs) > _MAX_REPORTED_PAIRS:
        text += f", and {len(pairs) - _MAX_REPORTED_PAIRS} more"
    return text


def check_updates(updates: Sequence[tuple[PanelInfo, PanelInfo]]) -> list[str]:
    """Return an error for each invalid update, given ``(base, head)`` panels."""
    errors = []
    for base, head in updates:
        name = head.path.name
        if (base.version.major, base.version.minor) != (
            head.version.major,
            head.version.minor,
        ):
            errors.append(
                f"{name}: version changed from {base.version} to {head.version}, "
                "but only the patch version can change in place. Add a new panel "
                "file for a new major or minor version"
            )
        elif head.version <= base.version:
            errors.append(
                f"{name}: version {head.version} must be greater than {base.version}"
            )
        added = set(head.sequences - base.sequences)
        removed = set(base.sequences - head.sequences)
        if added or removed:
            details = []
            if added:
                details.append(f"added {_format_pairs(added)}")
            if removed:
                details.append(f"removed {_format_pairs(removed)}")
            errors.append(
                f"{name}: sequences must not change in place ({'; '.join(details)})"
            )
    return errors


def _git(args: Sequence[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def modified_panels(
    base_ref: str, panel_dir: Path
) -> list[tuple[PanelInfo, PanelInfo]]:
    """Return ``(base, head)`` for each panel modified since the merge-base.

    The head panel is read from the working tree.
    """
    merge_base = _git(["merge-base", base_ref, "HEAD"], panel_dir).strip()
    names = _git(
        [
            "diff",
            "--name-only",
            "--diff-filter=M",
            "--no-renames",
            "--relative",
            merge_base,
            "--",
            ".",
        ],
        panel_dir,
    ).splitlines()
    updates = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(n for n in names if "/" not in n and n.endswith(".csv")):
            base_path = Path(tmp) / name
            base_path.write_text(_git(["show", f"{merge_base}:./{name}"], panel_dir))
            updates.append((load_panel(base_path), load_panel(panel_dir / name)))
    return updates


def check_panels(panel_dir: Path, base_ref: str) -> list[str]:
    """Run every update check on the panels modified since ``base_ref``."""
    return check_updates(modified_panels(base_ref, panel_dir))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the panels of the given directory, or of the bundled ones."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("panel_dir", nargs="?", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument(
        "--base",
        required=True,
        help="Git ref to compare with, for example origin/dev.",
    )
    args = parser.parse_args(argv)

    try:
        errors = check_panels(args.panel_dir, args.base)
    except subprocess.CalledProcessError as error:
        print(f"git failed: {error.stderr.strip()}")
        return 2
    if errors:
        print(f"Panel update checks failed for {args.panel_dir}:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"All panel updates in {args.panel_dir} passed the checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
