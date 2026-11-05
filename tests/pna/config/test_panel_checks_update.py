"""Tests for the checks on the bundled panel files modified since a base ref.

Copyright © 2026 Pixelgen Technologies AB.
"""

import subprocess
from pathlib import Path

import pytest

from tests.pna.config.panel_checks_helpers import MARKERS, update_checks, write_panel

# Same sequences as MARKERS, so the file differs without a sequence change.
_RENAMED_MARKERS = {"A2": MARKERS["A"], "B": MARKERS["B"]}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A git repo whose ``main`` branch holds one panel, checked out on ``feature``."""
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "panels").mkdir()
    write_panel(tmp_path / "panels", version="1.0.0")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "base")
    _git(tmp_path, "checkout", "-b", "feature")
    return tmp_path


def _update(repo: Path, **kwargs) -> list[str]:
    """Rewrite the panel, commit it and return the errors against ``main``."""
    write_panel(repo / "panels", **kwargs)
    _git(repo, "commit", "-am", "update")
    return update_checks.check_panels(repo / "panels", "main")


def test_unmodified_panel_passes(repo):
    assert update_checks.check_panels(repo / "panels", "main") == []


def test_patch_bump_with_same_sequences_passes(repo):
    assert _update(repo, version="1.0.1") == []


def test_marker_rename_and_reordering_pass(repo):
    markers = {"B2": MARKERS["B"], "A": MARKERS["A"]}

    assert _update(repo, version="1.0.1", markers=markers) == []


@pytest.mark.parametrize(
    "markers, expected",
    [
        ({"A": ("AAAA", "AAAA"), "B": ("CCCC", "GGGG")}, "added (CCCC, GGGG)"),
        ({"A": ("AAAA", "AAAA")}, "removed (CCCC, CCCC)"),
        ({**MARKERS, "C": ("TTTT", "TTTT")}, "added (TTTT, TTTT)"),
    ],
)
def test_changed_sequences_fail(repo, markers, expected):
    errors = _update(repo, version="1.0.1", markers=markers)

    assert len(errors) == 1
    assert "my-product-v1.0.csv" in errors[0]
    assert expected in errors[0]


@pytest.mark.parametrize("version", ["1.0.0", "1.0"])
def test_same_version_fails(repo, version):
    errors = _update(repo, version=version, markers=_RENAMED_MARKERS)

    assert any("must be greater than 1.0.0" in error for error in errors)


def test_lower_version_fails(repo):
    _git(repo, "checkout", "main")
    write_panel(repo / "panels", version="1.0.3")
    _git(repo, "commit", "-am", "bump")
    _git(repo, "checkout", "-B", "feature")

    errors = _update(repo, version="1.0.2", markers=_RENAMED_MARKERS)

    assert any("must be greater than 1.0.3" in error for error in errors)


def test_minor_bump_in_place_fails(repo):
    errors = _update(repo, version="1.1.0")

    assert any("only the patch version can change" in error for error in errors)


def test_uncommitted_change_is_checked(repo):
    write_panel(repo / "panels", version="1.0.0", markers={"A": MARKERS["A"]})

    errors = update_checks.check_panels(repo / "panels", "main")

    assert any("removed (CCCC, CCCC)" in error for error in errors)


def test_added_panel_is_skipped(repo):
    write_panel(repo / "panels", "my-product-v1.1", "1.1.0", {"X": ("GGGG", "GGGG")})
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "new panel")

    assert update_checks.check_panels(repo / "panels", "main") == []


def test_deleted_panel_is_skipped(repo):
    (repo / "panels" / "my-product-v1.0.csv").unlink()
    _git(repo, "commit", "-am", "delete")

    assert update_checks.check_panels(repo / "panels", "main") == []


def test_changes_before_the_merge_base_are_ignored(repo):
    # main moves on after the branch point; the branch itself is untouched.
    _git(repo, "checkout", "main")
    write_panel(repo / "panels", version="1.0.1", markers={"A": MARKERS["A"]})
    _git(repo, "commit", "-am", "main moves on")
    _git(repo, "checkout", "feature")

    assert update_checks.check_panels(repo / "panels", "main") == []


def test_main_exit_codes(repo):
    assert update_checks.main([str(repo / "panels"), "--base", "main"]) == 0

    _update(repo, version="1.0.1", markers={"A": MARKERS["A"]})
    assert update_checks.main([str(repo / "panels"), "--base", "main"]) == 1


def test_main_reports_git_failure(repo):
    assert update_checks.main([str(repo / "panels"), "--base", "missing-ref"]) == 2
