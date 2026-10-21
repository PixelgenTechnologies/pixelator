"""Copyright © 2025 Pixelgen Technologies AB."""

import click
import pytest

from pixelator.pna.cli.common import load_panels, validate_panel


def test_validate_panel():
    """Verify validate panel."""
    panel = "proxiome-v1-immuno-155-v1.0"
    assert validate_panel(None, None, panel) == panel


def test_validate_custom_panel(pna_data_root):
    """Verify validate custom panel.

    Args:
        pna_data_root: pna data root.
    """
    panel_file = pna_data_root / "panels/test-pna-panel.csv"
    assert validate_panel(None, None, panel_file) == panel_file


def test_validate_non_existing_panel():
    """Test that validating a non-existing panel raises a UsageError."""
    panel = "non-existing-panel"
    with pytest.raises(click.UsageError):
        validate_panel(None, None, panel)


def test_validate_panel_accepts_repeated_values(pna_data_root):
    """Verify that repeated panel values are validated one by one."""
    name = "proxiome-v1-immuno-155-v1.0"
    panel_file = pna_data_root / "panels/test-pna-panel.csv"
    assert validate_panel(None, None, (name, panel_file)) == (name, panel_file)


def test_validate_panel_rejects_a_bad_repeated_value():
    """Verify that one unknown panel among several raises a UsageError."""
    with pytest.raises(click.UsageError):
        validate_panel(None, None, ("proxiome-v1-immuno-155-v1.0", "non-existing"))


def _write_panel_csv(path, name, rows):
    header = f"# ---\n# name: {name}\n# product: kit\n# version: 1.0.0\n# ---\n"
    lines = ["marker_id,control,sequence_1,sequence_2"]
    lines += [f"{marker},,{sequence},{sequence}" for marker, sequence in rows]
    path.write_text(header + "\n".join(lines) + "\n")
    return path


def test_load_panels_concatenates_the_panels(tmp_path):
    left = _write_panel_csv(tmp_path / "left.csv", "left", [("CD3", "AAAA")])
    right = _write_panel_csv(tmp_path / "right.csv", "right", [("CD19", "CCCC")])

    panel = load_panels((left, right))

    assert panel.markers == ["CD3", "CD19"]


def test_load_panels_turns_overlapping_panels_into_a_usage_error(tmp_path):
    left = _write_panel_csv(tmp_path / "left.csv", "left", [("CD3", "AAAA")])
    right = _write_panel_csv(tmp_path / "right.csv", "right", [("CD3", "CCCC")])

    with pytest.raises(click.UsageError, match=r"(?s)--panel.*marker_id 'CD3'"):
        load_panels((left, right))
