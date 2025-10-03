"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

from pandas.testing import assert_frame_equal
import pytest

from pixelator.pna.analysis_engine import AnalysisManager
from pixelator.pna.layout import CreateLayout
from pixelator.pna.pixeldataset import PixelDatasetSaver, PNAPixelDataset


@pytest.mark.slow
def test_layout(pna_pxl_dataset: PNAPixelDataset, tmp_path):
    manager = AnalysisManager(
        [
            CreateLayout(
                ["wpmds_3d"],
            )
        ]
    )

    pna_pixel_filtered = pna_pxl_dataset.filter(
        components=pna_pxl_dataset.adata().obs.index[:2]
    )
    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pixel_filtered).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    dataset = manager.execute(pna_pixel_filtered, pxl_file_target)

    result = dataset.precomputed_layouts(add_marker_counts=False).to_df()
    assert sorted(result.columns.to_list()) == sorted(
        [
            "sample",
            "component",
            "graph_projection",
            "index",
            "layout",
            "pixel_type",
            "x",
            "y",
            "z",
        ]
    )
    assert "A" in set(result["pixel_type"].to_list())
    assert "B" in set(result["pixel_type"].to_list())


@pytest.mark.slow
def test_layout_from_path(pna_pxl_dataset: PNAPixelDataset, tmp_path):
    manager = AnalysisManager(
        [
            CreateLayout(
                ["wpmds_3d"],
            )
        ]
    )

    pna_pixel_filtered = pna_pxl_dataset.filter(
        components=pna_pxl_dataset.adata().obs.index[:2]
    )

    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pixel_filtered).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )

    dataset = manager.execute_from_path(pxl_file_target.path, pxl_file_target)
    result = dataset.precomputed_layouts(add_marker_counts=False).to_df()
    assert sorted(result.columns.to_list()) == sorted(
        [
            "sample",
            "component",
            "graph_projection",
            "index",
            "layout",
            "pixel_type",
            "x",
            "y",
            "z",
        ]
    )
    assert "A" in set(result["pixel_type"].to_list())
    assert "B" in set(result["pixel_type"].to_list())
