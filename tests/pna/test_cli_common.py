"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.cli.common import validate_panel


def test_validate_panel():
    panel = "proxiome-immuno-155"
    assert validate_panel(None, None, panel) == panel
