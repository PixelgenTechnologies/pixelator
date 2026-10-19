"""Tests for aligning the marker names of pxl files with different panel patches.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset.io import (
    PixelDataViewer,
    PixelFileWriter,
    PxlFile,
    Query,
)
from pixelator.pna.pixeldataset.io.pixel_data_viewer import _marker_rename_expression


def _marker(marker_id: str, sequence: str) -> dict:
    return {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
    }


def _panel(
    version: str, markers: dict[str, str], name: str = "base", product: str = "kit"
) -> PNAAntibodyPanel:
    """Build a panel from ``marker id -> sequence``."""
    rows = [_marker(marker_id, sequence) for marker_id, sequence in markers.items()]
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(name=name, version=version, product=product),
    )


def _write_pxl(
    path: Path,
    sample: str,
    panel: PNAAntibodyPanel | None,
    marker_pair: tuple[str, str],
    legacy_columns: bool = False,
) -> PxlFile:
    """Write a pxl file whose single edge and proximity row use ``marker_pair``."""
    marker_1, marker_2 = marker_pair
    edgelist = pl.DataFrame(
        {
            "umi1": [1],
            "umi2": [2],
            "read_count": [1],
            "uei_count": [1],
            "marker_1": [marker_1],
            "marker_2": [marker_2],
            "component": ["c1"],
        },
        schema={
            "umi1": pl.UInt64,
            "umi2": pl.UInt64,
            "read_count": pl.UInt32,
            "uei_count": pl.UInt32,
            "marker_1": pl.Utf8,
            "marker_2": pl.Utf8,
            "component": pl.Utf8,
        },
    )
    proximity = pl.DataFrame(
        {"marker_1": [marker_1], "marker_2": [marker_2], "component": ["c1"]}
    )
    with PixelFileWriter(path) as writer:
        writer.write_edgelist(edgelist)
        connection = writer.get_connection()
        if panel is not None:
            adata = pna_edgelist_to_anndata(connection, panel=panel)
        else:
            adata = anndata.AnnData(
                X=np.zeros((1, 2)),
                obs=pd.DataFrame(index=["c1"]),
                var=pd.DataFrame(index=pd.Index(marker_pair, name="marker_id")),
            )
        writer.write_adata(adata)
        writer.write_metadata({"sample_name": sample})
        writer.write_proximity(proximity)
        if legacy_columns:
            connection.execute("ALTER TABLE edgelist RENAME COLUMN marker_1 TO marker1")
    return PxlFile(path)


def _viewer(*pxl_files: PxlFile) -> PixelDataViewer:
    return PixelDataViewer.from_files(list(pxl_files))


def _markers(viewer: PixelDataViewer, table: str = "edgelist") -> dict[str, set[str]]:
    """Return the marker ids of each sample in a table of the view."""
    with viewer.open() as session:
        df = session.execute_eager(
            Query(f"SELECT marker_1, marker_2, sample FROM {table}", {})
        )
    return {
        sample: set(group["marker_1"]) | set(group["marker_2"])
        for (sample,), group in df.group_by("sample")
    }


@pytest.fixture(name="old_file")
def old_file_fixture(tmp_path) -> PxlFile:
    panel = _panel("1.0.0", {"CD3": "AAAA", "CD4": "CCCC"})
    return _write_pxl(tmp_path / "old.pxl", "old", panel, ("CD3", "CD4"))


@pytest.fixture(name="new_file")
def new_file_fixture(tmp_path) -> PxlFile:
    panel = _panel("1.0.1", {"CD3E": "AAAA", "CD4": "CCCC"})
    return _write_pxl(tmp_path / "new.pxl", "new", panel, ("CD3E", "CD4"))


class TestReadPanel:
    """Reading the panel stored in a pxl file."""

    def test_returns_the_panel_stored_in_the_file(self, old_file):
        assert old_file.read_panel() == _panel("1.0.0", {"CD3": "AAAA", "CD4": "CCCC"})

    def test_returns_none_when_the_file_stores_no_panel(self, tmp_path):
        pxl_file = _write_pxl(tmp_path / "x.pxl", "x", None, ("CD3", "CD4"))

        assert pxl_file.read_panel() is None


class TestMarkerAlignment:
    """Marker names in a view over files with different panel patches."""

    def test_edgelist_and_proximity_use_the_newest_marker_names(
        self, old_file, new_file
    ):
        viewer = _viewer(old_file, new_file)

        for table in ("edgelist", "proximity"):
            assert _markers(viewer, table) == {
                "old": {"CD3E", "CD4"},
                "new": {"CD3E", "CD4"},
            }

    def test_marker_renames_lists_the_samples_that_change(self, old_file, new_file):
        assert _viewer(old_file, new_file).marker_renames() == {"old": {"CD3": "CD3E"}}

    def test_legacy_marker_columns_are_renamed(self, tmp_path, new_file):
        panel = _panel("1.0.0", {"CD3": "AAAA", "CD4": "CCCC"})
        legacy = _write_pxl(
            tmp_path / "legacy.pxl", "old", panel, ("CD3", "CD4"), legacy_columns=True
        )

        with _viewer(legacy, new_file).open() as session:
            df = session.execute_eager(
                Query("SELECT marker1 FROM edgelist WHERE sample = 'old'", {})
            )

        assert df["marker1"].to_list() == ["CD3E"]

    def test_files_with_identical_panels_are_left_alone(self, old_file, tmp_path):
        panel = _panel("1.0.0", {"CD3": "AAAA", "CD4": "CCCC"})
        same = _write_pxl(tmp_path / "same.pxl", "same", panel, ("CD3", "CD4"))

        viewer = _viewer(old_file, same)

        assert viewer.marker_renames() == {}
        assert _markers(viewer)["same"] == {"CD3", "CD4"}

    def test_a_single_file_does_not_read_its_panel(self, old_file, monkeypatch):
        def fail(self):
            raise AssertionError("the panel should not be read")

        monkeypatch.setattr(PxlFile, "read_panel", fail)

        assert _viewer(old_file).marker_renames() == {}

    def test_different_addons_only_rename_the_shared_base(self, tmp_path, monkeypatch):
        base_old = _panel("1.0.0", {"CD3": "AAAA"}, name="base", product="base")
        base_new = _panel("1.0.1", {"CD3E": "AAAA"}, name="base", product="base")
        addon_a = _panel("1.0.0", {"CD19": "CCCC"}, name="a", product="addon_a")
        addon_b = _panel("1.0.0", {"CD20": "GGGG"}, name="b", product="addon_b")
        # Files cannot store a panel with several sources yet, so they store
        # the base panel and the combined panels are served by read_panel.
        panels = {
            "a.pxl": PNAAntibodyPanel.concatenate([base_old, addon_a]),
            "b.pxl": PNAAntibodyPanel.concatenate([base_new, addon_b]),
        }
        monkeypatch.setattr(PxlFile, "read_panel", lambda self: panels[self.path.name])
        sample_a = _write_pxl(tmp_path / "a.pxl", "a", base_old, ("CD3", "CD19"))
        sample_b = _write_pxl(tmp_path / "b.pxl", "b", base_new, ("CD3E", "CD20"))

        viewer = _viewer(sample_a, sample_b)

        assert viewer.marker_renames() == {"a": {"CD3": "CD3E"}}
        assert _markers(viewer) == {"a": {"CD3E", "CD19"}, "b": {"CD3E", "CD20"}}

    def test_filtering_samples_keeps_the_aligned_names(self, old_file, new_file):
        filtered = _viewer(old_file, new_file).filter_samples({"old"})

        assert _markers(filtered) == {"old": {"CD3E", "CD4"}}

    def test_unalignable_panels_leave_the_markers_unchanged(self, old_file, tmp_path):
        other = _panel("1.1.0", {"CD3E": "AAAA", "CD4": "CCCC"})
        other_file = _write_pxl(tmp_path / "o.pxl", "other", other, ("CD3E", "CD4"))

        viewer = _viewer(old_file, other_file)

        assert viewer.marker_renames() == {}
        assert _markers(viewer) == {"old": {"CD3", "CD4"}, "other": {"CD3E", "CD4"}}

    def test_a_file_without_a_panel_leaves_the_markers_unchanged(
        self, old_file, tmp_path
    ):
        no_panel = _write_pxl(tmp_path / "n.pxl", "none", None, ("CD3E", "CD4"))

        viewer = _viewer(old_file, no_panel)

        assert viewer.marker_renames() == {}
        assert _markers(viewer) == {"old": {"CD3", "CD4"}, "none": {"CD3E", "CD4"}}


def test_marker_ids_with_quotes_are_escaped_in_the_rename_expression():
    expression = _marker_rename_expression("marker_1", {"A'B": "C"})

    assert "WHEN 'A''B' THEN 'C'" in expression
