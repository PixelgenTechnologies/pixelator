"""Tests for storing a panel in the tables of a pxl file.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.config.panel.tables import (
    has_panel_tables,
    panel_tables_present,
    read_panel_table_frames,
)
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile


def _panel(name: str, version: str, rows: list[dict], **metadata) -> PNAAntibodyPanel:
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(
            name=name, version=version, product="proxiome", **metadata
        ),
        file_name=f"{name}.csv",
    )


def _marker(marker_id: str, sequence: str, **extra) -> dict:
    return {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
        **extra,
    }


def _store_and_load(path: Path, panel: PNAAntibodyPanel) -> PNAAntibodyPanel:
    with PixelFileWriter(path) as writer:
        writer.write_panel(panel)
        frames = read_panel_table_frames(writer.get_connection())
    assert frames is not None
    return PNAAntibodyPanel.from_panel_tables(*frames)


def test_a_single_source_panel_round_trips(tmp_path: Path):
    panel = _panel(
        "base",
        "1.0.0",
        [
            _marker("CD3", "AAAA", uniprot_id="P01730"),
            _marker("CD4", "CCCC", uniprot_id="P01732"),
        ],
        description="A base panel",
        aliases=["b"],
    )

    loaded = _store_and_load(tmp_path / "sample.pxl", panel)

    assert loaded == panel
    assert loaded.filename == "base.csv"
    assert loaded.metadata.aliases == ["b"]
    assert list(loaded.markers) == ["CD3", "CD4"]


def test_a_multi_source_panel_round_trips_with_its_sources(tmp_path: Path):
    base = _panel("base", "1.0.0", [_marker("CD3", "AAAA", uniprot_id="P01730")])
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")])
    panel = PNAAntibodyPanel.concatenate([base, addon])

    loaded = _store_and_load(tmp_path / "sample.pxl", panel)

    assert loaded == panel
    assert [source.metadata.name for source in loaded.sources] == ["base", "addon"]
    assert [source.columns for source in loaded.sources] == [
        source.columns for source in panel.sources
    ]
    assert list(loaded.marker_source_ids) == [0, 1]
    assert loaded.filename is None


def test_hashing_clones_round_trip(tmp_path: Path):
    panel = _panel(
        "base",
        "1.0.0",
        [
            _marker("CD3", "AAAA"),
            _marker("B2M-1", "CCCC", sample_hashing="yes"),
            _marker("B2M-2", "GGGG", sample_hashing="yes"),
        ],
    )

    loaded = _store_and_load(tmp_path / "sample.pxl", panel)

    assert loaded == panel
    assert sorted(loaded.hashing_marker_ids) == ["B2M-1", "B2M-2"]


def test_writing_a_panel_again_replaces_the_stored_one(tmp_path: Path):
    first = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    second = _panel("base", "1.0.1", [_marker("CD3E", "AAAA")])
    path = tmp_path / "sample.pxl"

    with PixelFileWriter(path) as writer:
        writer.write_panel(first)
        writer.write_panel(second)
        frames = read_panel_table_frames(writer.get_connection())

    assert frames is not None
    assert PNAAntibodyPanel.from_panel_tables(*frames) == second


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["panels", "panel_sources", "edgelist"], True),
        (["panels", "edgelist"], False),
        (["panel_sources"], False),
        ([], False),
    ],
)
def test_has_panel_tables_needs_both_tables(names: list[str], expected: bool):
    assert has_panel_tables(names) is expected


def test_a_file_without_panel_tables_has_no_frames(tmp_path: Path):
    with PixelFileWriter(tmp_path / "sample.pxl") as writer:
        connection = writer.get_connection()
        connection.execute("CREATE TABLE edgelist (umi1 INTEGER)")

        assert not panel_tables_present(connection)
        assert read_panel_table_frames(connection) is None


@pytest.mark.parametrize("table", ["panels", "panel_sources"])
def test_both_panel_tables_are_written(tmp_path: Path, table: str):
    with PixelFileWriter(tmp_path / "sample.pxl") as writer:
        writer.write_panel(_panel("base", "1.0.0", [_marker("CD3", "AAAA")]))
        names = writer.get_connection().execute("SHOW TABLES").fetchdf()["name"]

    assert table in set(names)


def _adata(markers: list[str]) -> AnnData:
    return AnnData(
        X=np.zeros((1, len(markers))),
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(
            {"antibody_count": 1}, index=pd.Index(markers, name="marker_id")
        ),
    )


def _write_pxl(path: Path, panel: PNAAntibodyPanel | None, markers: list[str]) -> None:
    with PixelFileWriter(path) as writer:
        writer.write_metadata({"sample_name": path.stem})
        if panel is not None:
            writer.write_panel(panel)
        writer.get_connection().execute("CREATE TABLE edgelist (umi1 INTEGER)")
        writer.write_adata(_adata(markers))


def test_read_panel_returns_the_stored_panel_with_all_its_sources(tmp_path: Path):
    base = _panel("base", "1.0.0", [_marker("CD3", "AAAA", uniprot_id="P01730")])
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")])
    panel = PNAAntibodyPanel.concatenate([base, addon])
    path = tmp_path / "sample.pxl"
    _write_pxl(path, panel, ["CD3", "CD19"])

    assert PxlFile(path).read_panel() == panel


def test_from_pxl_dataset_returns_the_stored_panel(tmp_path: Path):
    panel = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    _write_pxl(tmp_path / "a.pxl", panel, ["CD3"])
    _write_pxl(tmp_path / "b.pxl", panel, ["CD3"])

    dataset = read([tmp_path / "a.pxl", tmp_path / "b.pxl"])

    assert PNAAntibodyPanel.from_pxl_dataset(dataset) == panel


def test_from_pxl_dataset_raises_key_error_when_a_file_has_no_panel(tmp_path: Path):
    path = tmp_path / "old.pxl"
    _write_pxl(path, None, ["CD3"])

    with pytest.raises(KeyError, match="panel information"):
        PNAAntibodyPanel.from_pxl_dataset(read(path))


def test_from_pxl_dataset_rejects_files_with_different_panels(tmp_path: Path):
    first = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    second = _panel("base", "1.0.1", [_marker("CD3E", "AAAA")])
    _write_pxl(tmp_path / "a.pxl", first, ["CD3"])
    _write_pxl(tmp_path / "b.pxl", second, ["CD3E"])

    dataset = read([tmp_path / "a.pxl", tmp_path / "b.pxl"])

    with pytest.raises(ValueError, match="do not store the same panel"):
        PNAAntibodyPanel.from_pxl_dataset(dataset)


def test_stored_panel_columns_are_joined_back_onto_var(tmp_path: Path):
    panel = _panel("base", "1.0.0", [_marker("CD3", "AAAA", uniprot_id="P01730")])
    path = tmp_path / "sample.pxl"
    _write_pxl(path, panel, ["CD3"])

    adata = read(path).adata(add_clr_transform=False, add_log1p_transform=False)

    assert adata.var.loc["CD3", "uniprot_id"] == "P01730"
    assert adata.var.loc["CD3", "sequence_1"] == "AAAA"
    assert "panel_metadata" not in adata.uns
