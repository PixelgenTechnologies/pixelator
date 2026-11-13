"""Helpers for testing the panel check scripts in ``utils``.

Copyright © 2026 Pixelgen Technologies AB.
"""

import sys
from pathlib import Path

# The scripts live next to each other in a folder that is not a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "utils"))

import check_panels_static as static_checks  # noqa: E402
import check_panels_update as update_checks  # noqa: E402
import panel_files  # noqa: E402

__all__ = [
    "MARKERS",
    "STEM",
    "panel_files",
    "static_checks",
    "update_checks",
    "write_panel",
]

MARKERS = {"A": ("AAAA", "AAAA"), "B": ("CCCC", "CCCC")}
STEM = "my-product-v1.0"


def write_panel(
    directory: Path,
    stem: str = STEM,
    version: str = "1.0.0",
    markers: dict[str, tuple[str, str]] = MARKERS,
    product: str | None = "my-product",
    name: str | None = None,
) -> None:
    """Write a panel CSV named after ``stem`` into ``directory``."""
    header = ["# ---", f"# name: {name or stem}"]
    if product:
        header.append(f"# product: {product}")
    header += [f'# version: "{version}"', "# ---"]
    rows = [
        f"{marker_id},,{first},{second}"
        for marker_id, (first, second) in markers.items()
    ]
    (directory / f"{stem}.csv").write_text(
        "\n".join([*header, "marker_id,control,sequence_1,sequence_2", *rows]) + "\n"
    )
