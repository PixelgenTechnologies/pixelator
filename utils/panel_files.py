"""Read the bundled PNA panel files for the panel check scripts.

Copyright © 2026 Pixelgen Technologies AB.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


@dataclass(frozen=True)
class PanelInfo:
    """The parts of a panel file that the checks need."""

    path: Path
    name: str
    product: Optional[str]
    version: Version
    sequences: frozenset[tuple[str, str]]

    @property
    def product_key(self) -> str:
        """Identify the product, falling back on the name."""
        return self.product or self.name


def load_panel(path: Path) -> PanelInfo:
    """Read the header and sequence pairs of a panel CSV."""
    metadata = parse_panel_header_metadata(path)
    table = pd.read_csv(path, comment="#", usecols=_SEQUENCE_COLUMNS, dtype=str).fillna(
        ""
    )
    return PanelInfo(
        path=path,
        name=metadata.name,
        product=metadata.product,
        version=Version(metadata.version),
        sequences=frozenset(
            tuple(row) for row in table[_SEQUENCE_COLUMNS].itertuples(index=False)
        ),
    )


def load_panels(panel_dir: Path) -> list[PanelInfo]:
    """Read every panel CSV in a directory."""
    return [load_panel(path) for path in sorted(panel_dir.glob("*.csv"))]
