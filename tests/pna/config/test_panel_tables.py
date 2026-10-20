"""Tests for storing a panel in the tables of a pxl file.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import pandas as pd
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.config.panel.tables import (
    panel_tables_present,
    read_panel_table_frames,
)
from pixelator.pna.pixeldataset.io import PixelFileWriter


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
