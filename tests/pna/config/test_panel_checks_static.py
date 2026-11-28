"""Tests for the static checks on the bundled panel files.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pytest

from tests.pna.config.panel_checks_helpers import (
    panel_files,
    static_checks,
    write_panel,
)


def test_unique_versions_same_product_and_version_fails(tmp_path):
    write_panel(tmp_path, "my-product-v1.0", "1.0.0")
    write_panel(tmp_path, "my-product-prerelease", "1.0.0")

    errors = static_checks.check_panels(tmp_path)

    assert any(
        "my-product-prerelease.csv, my-product-v1.0.csv" in error
        or "my-product-v1.0.csv, my-product-prerelease.csv" in error
        for error in errors
    )


def test_unique_versions_equivalent_version_strings_fail(tmp_path):
    write_panel(tmp_path, "my-product-v1.0", "1.0")
    write_panel(tmp_path, "my-product-prerelease", "1.0.0")

    assert static_checks.check_unique_versions(panel_files.load_panels(tmp_path))


def test_unique_versions_same_version_in_other_product_passes(tmp_path):
    write_panel(tmp_path, "first-v1.0", product="first")
    write_panel(tmp_path, "second-v1.0", product="second")

    assert static_checks.check_panels(tmp_path) == []


def test_unique_versions_product_defaults_to_name(tmp_path):
    # "one-v1.0" has no product, so its name stands in for it.
    write_panel(tmp_path, "one-v1.0", product=None)
    write_panel(tmp_path, "one-v1.0-prerelease", product="one-v1.0")

    panels = panel_files.load_panels(tmp_path)

    assert static_checks.check_unique_versions(panels)


def test_filenames_valid_panels_pass(tmp_path):
    write_panel(tmp_path, "my-product-v1.0", "1.0.5")
    write_panel(tmp_path, "my-product-v2.1", "2.1.0")
    write_panel(tmp_path, "my-product-prerelease", "0.3.1")

    assert static_checks.check_panels(tmp_path) == []


def test_filenames_name_must_match_file_name(tmp_path):
    write_panel(tmp_path, "my-product-v1.0", name="my-product-v1.1")

    errors = static_checks.check_panels(tmp_path)

    assert len(errors) == 1
    assert "name 'my-product-v1.1' must be 'my-product-v1.0'" in errors[0]


def test_filenames_product_must_prefix_file_name(tmp_path):
    write_panel(tmp_path, "my-product-v1.0", product="other-product")

    errors = static_checks.check_panels(tmp_path)

    assert len(errors) == 1
    assert "must start with the product 'other-product'" in errors[0]


def test_filenames_missing_product_is_exempt(tmp_path):
    write_panel(tmp_path, "anything-v1.0", product=None)

    assert static_checks.check_panels(tmp_path) == []


@pytest.mark.parametrize("version", ["1.1.0", "2.0.0", "0.9.9"])
def test_filenames_version_must_match_file_name(tmp_path, version):
    write_panel(tmp_path, "my-product-v1.0", version)

    errors = static_checks.check_panels(tmp_path)

    assert len(errors) == 1
    assert "must start with 1.0" in errors[0]


def test_filenames_prerelease_version_is_free(tmp_path):
    write_panel(tmp_path, "my-product-prerelease", "7.8.9")

    assert static_checks.check_panels(tmp_path) == []


def test_filenames_unknown_suffix_fails(tmp_path):
    write_panel(tmp_path, "my-product-final")

    errors = static_checks.check_panels(tmp_path)

    assert len(errors) == 1
    assert "must end with -v<major>.<minor> or -prerelease" in errors[0]


def test_main_exit_codes(tmp_path):
    write_panel(tmp_path)
    assert static_checks.main([str(tmp_path)]) == 0

    write_panel(tmp_path, "my-product-prerelease", "1.0.0")
    assert static_checks.main([str(tmp_path)]) == 1


def test_bundled_panels_pass():
    assert static_checks.check_panels(panel_files.DEFAULT_PANEL_DIR) == []
