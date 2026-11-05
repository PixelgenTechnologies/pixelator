#!/usr/bin/env python3
"""Check that the bundled PNA panels have consistent metadata.

Copyright © 2026 Pixelgen Technologies AB.

The checks only need the panel files as they are, and run on every panel:

1. No two panels share both a product and a version.
2. The panel ``name`` is the file name without its extension.
3. The ``product``, when set, is a prefix of the file name.
4. The file name ends in ``-v{major}.{minor}`` and the version has that major
   and minor, or it ends in ``-prerelease`` and the version is free.

A panel without a product is identified by its name, as in ``align_panels``.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from packaging.version import Version
from panel_files import DEFAULT_PANEL_DIR, PanelInfo, load_panels

_VERSIONED_STEM = re.compile(r"-v(?P<major>\d+)\.(?P<minor>\d+)$")
_PRERELEASE_SUFFIX = "-prerelease"


def check_unique_versions(panels: Sequence[PanelInfo]) -> list[str]:
    """Return an error for each product and version found in several files."""
    seen: dict[tuple[str, Version], list[str]] = {}
    for panel in panels:
        seen.setdefault((panel.product_key, panel.version), []).append(panel.path.name)
    return [
        f"Product {product!r} has version {version} in several files: "
        + ", ".join(names)
        for (product, version), names in seen.items()
        if len(names) > 1
    ]


def check_filenames(panels: Sequence[PanelInfo]) -> list[str]:
    """Return an error for each panel whose metadata does not match its file name."""
    errors = []
    for panel in panels:
        stem = panel.path.stem
        if panel.name != stem:
            errors.append(f"{panel.path.name}: name {panel.name!r} must be {stem!r}")
        if panel.product and not stem.startswith(panel.product):
            errors.append(
                f"{panel.path.name}: file name must start with the product "
                f"{panel.product!r}"
            )
        versioned = _VERSIONED_STEM.search(stem)
        if versioned:
            expected = (int(versioned["major"]), int(versioned["minor"]))
            if (panel.version.major, panel.version.minor) != expected:
                errors.append(
                    f"{panel.path.name}: version {panel.version} must start with "
                    f"{expected[0]}.{expected[1]} to match the file name"
                )
        elif not stem.endswith(_PRERELEASE_SUFFIX):
            errors.append(
                f"{panel.path.name}: file name must end with -v<major>.<minor> "
                f"or {_PRERELEASE_SUFFIX}"
            )
    return errors


def check_panels(panel_dir: Path) -> list[str]:
    """Run every static check on the panels of a directory."""
    panels = load_panels(panel_dir)
    return check_unique_versions(panels) + check_filenames(panels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check the panels of the given directory, or of the bundled ones."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("panel_dir", nargs="?", type=Path, default=DEFAULT_PANEL_DIR)
    args = parser.parse_args(argv)

    errors = check_panels(args.panel_dir)
    if errors:
        print(f"Panel checks failed for {args.panel_dir}:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"All panels in {args.panel_dir} passed the static checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
