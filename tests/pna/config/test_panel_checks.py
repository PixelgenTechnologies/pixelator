"""Tests for the CI checks on the bundled panel files.

Copyright © 2026 Pixelgen Technologies AB.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "utils" / "panel_checks.py"
_spec = importlib.util.spec_from_file_location("panel_checks", _SCRIPT)
panel_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel_checks)


def _write_panel(
    directory: Path,
    file_name: str,
    version: str,
    markers: dict[str, tuple[str, str]],
    product: str | None = "my-product",
) -> None:
    header = ["# ---", f"# name: {file_name.removesuffix('.csv')}"]
    if product:
        header.append(f"# product: {product}")
    header += [f'# version: "{version}"', "# ---"]
    rows = [
        f"{marker_id},,{first},{second}"
        for marker_id, (first, second) in markers.items()
    ]
    (directory / file_name).write_text(
        "\n".join([*header, "marker_id,control,sequence_1,sequence_2", *rows]) + "\n"
    )


_MARKERS = {"A": ("AAAA", "AAAA"), "B": ("CCCC", "CCCC")}


class TestUniqueVersions:
    def test_same_product_and_version_fails(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", "1.0.0", _MARKERS)

        errors = panel_checks.check_panels(tmp_path)

        assert any("one.csv, two.csv" in error for error in errors)

    def test_equivalent_version_strings_fail(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", "1.0.0", _MARKERS)

        assert panel_checks.check_unique_versions(panel_checks.load_panels(tmp_path))

    def test_same_version_in_other_product_passes(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS, product="first")
        _write_panel(tmp_path, "two.csv", "1.0.0", _MARKERS, product="second")

        assert panel_checks.check_panels(tmp_path) == []

    def test_product_defaults_to_name(self, tmp_path):
        # "one" has no product, so its name stands in for it.
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS, product=None)
        _write_panel(tmp_path, "two.csv", "1.0.0", _MARKERS, product="one")

        errors = panel_checks.check_panels(tmp_path)

        assert any("one.csv, two.csv" in error for error in errors)


class TestPatchSequences:
    def test_unchanged_sequences_pass(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", "1.0.1", _MARKERS)

        assert panel_checks.check_panels(tmp_path) == []

    def test_renamed_marker_passes(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(
            tmp_path, "two.csv", "1.0.1", {"A": _MARKERS["A"], "B2": _MARKERS["B"]}
        )

        assert panel_checks.check_panels(tmp_path) == []

    @pytest.mark.parametrize(
        "markers, expected",
        [
            ({"A": ("AAAA", "AAAA"), "B": ("CCCC", "GGGG")}, "added (CCCC, GGGG)"),
            ({"A": ("AAAA", "AAAA")}, "removed (CCCC, CCCC)"),
            ({**_MARKERS, "C": ("TTTT", "TTTT")}, "added (TTTT, TTTT)"),
        ],
    )
    def test_changed_sequences_fail(self, tmp_path, markers, expected):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", "1.0.1", markers)

        errors = panel_checks.check_panels(tmp_path)

        assert len(errors) == 1
        assert "two.csv" in errors[0]
        assert expected in errors[0]

    def test_compares_with_oldest_patch(self, tmp_path):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", "1.0.1", _MARKERS)
        _write_panel(tmp_path, "three.csv", "1.0.2", {"A": _MARKERS["A"]})

        errors = panel_checks.check_panels(tmp_path)

        assert len(errors) == 1
        assert "three.csv" in errors[0]

    @pytest.mark.parametrize("new_version", ["1.1.0", "2.0.0"])
    def test_minor_or_major_bump_may_change_sequences(self, tmp_path, new_version):
        _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
        _write_panel(tmp_path, "two.csv", new_version, {"A": ("TTTT", "TTTT")})

        assert panel_checks.check_panels(tmp_path) == []


def test_main_exit_codes(tmp_path):
    _write_panel(tmp_path, "one.csv", "1.0.0", _MARKERS)
    assert panel_checks.main([str(tmp_path)]) == 0

    _write_panel(tmp_path, "two.csv", "1.0.0", _MARKERS)
    assert panel_checks.main([str(tmp_path)]) == 1


def test_bundled_panels_pass():
    assert panel_checks.check_panels(panel_checks.DEFAULT_PANEL_DIR) == []
