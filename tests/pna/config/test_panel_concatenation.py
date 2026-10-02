"""Tests for concatenating several antibody panels into one.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas.testing import assert_frame_equal

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel, align_panel_patches
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile, read_dataset_panel


def _panel(
    name: str, version: str, rows: list[dict], product: str = "proxiome"
) -> PNAAntibodyPanel:
    frame = pd.DataFrame(rows).set_index("marker_id")
    return PNAAntibodyPanel(
        frame,
        AntibodyPanelMetadata(name=name, version=version, product=product),
        file_name=f"{name}.csv",
    )


def _marker(marker_id: str, sequence: str, **extra) -> dict:
    row = {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
    }
    row.update(extra)
    return row


def test_concatenate_keeps_metadata_for_each_source():
    base = _panel(
        "base",
        "1.0.0",
        [_marker("CD3", "AAAA")],
    )
    addon = _panel(
        "addon",
        "1.2.0",
        [_marker("CD19", "CCCC")],
    )

    combined = PNAAntibodyPanel.concatenate([base, addon])

    for field in (
        "metadata",
        "name",
        "version",
        "product",
        "description",
        "aliases",
        "archived",
    ):
        with pytest.raises(ValueError, match="single source"):
            getattr(combined, field)
    assert [source.metadata.name for source in combined.sources] == ["base", "addon"]
    assert [source.metadata.version for source in combined.sources] == [
        "1.0.0",
        "1.2.0",
    ]
    assert list(combined.markers) == ["CD3", "CD19"]
    assert combined.marker_source_ids.loc["CD3"] == 0
    assert combined.marker_source_ids.loc["CD19"] == 1


def test_equality_ignores_row_order_and_filename_but_not_marker_source():
    left = _panel("base", "1.0.0", [_marker("CD3", "AAAA"), _marker("CD19", "CCCC")])
    right = PNAAntibodyPanel(
        left.df.iloc[::-1],
        AntibodyPanelMetadata(name="base", version="1.0.0", product="proxiome"),
        file_name="other.csv",
    )
    assert left == right

    base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    addon = _panel("addon", "1.2.0", [_marker("CD19", "CCCC")])
    frame = pd.concat([base.df, addon.df])
    frame.index.name = "marker_id"
    sources = [*base.sources, *addon.sources]
    assigned = PNAAntibodyPanel(
        frame,
        sources=sources,
        marker_source_ids=pd.Series({"CD3": 0, "CD19": 1}, dtype="int64"),
    )
    swapped = PNAAntibodyPanel(
        frame,
        sources=sources,
        marker_source_ids=pd.Series({"CD3": 1, "CD19": 0}, dtype="int64"),
    )
    assert assigned != swapped


def test_concatenate_one_panel_keeps_its_metadata():
    panel = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    assert PNAAntibodyPanel.concatenate([panel]) is panel
    assert panel.name == "base"


def test_concatenate_blanks_optional_columns_missing_from_one_source():
    with_uniprot = _panel(
        "base",
        "1.0.0",
        [_marker("CD3", "AAAA", uniprot_id="P01730")],
    )
    without_uniprot = _panel("addon", "1.0.0", [_marker("CD19", "CCCC")])

    combined = PNAAntibodyPanel.concatenate([with_uniprot, without_uniprot])

    assert combined.df.loc["CD3", "uniprot_id"] == "P01730"
    assert combined.df.loc["CD19", "uniprot_id"] == ""

    hashing = _panel(
        "hashing",
        "1.0.0",
        [_marker("B2M-1", "GGGG", sample_hashing=True)],
    )
    combined_hashing = PNAAntibodyPanel.concatenate([without_uniprot, hashing])
    assert combined_hashing.hashing_marker_ids == {"B2M-1"}


def test_concatenate_rejects_duplicate_marker_and_sequence():
    left = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    same_marker = _panel("addon", "1.0.0", [_marker("CD3", "CCCC")])
    with pytest.raises(AssertionError, match="marker_id were not unique"):
        PNAAntibodyPanel.concatenate([left, same_marker])

    same_sequence = _panel("addon", "1.0.0", [_marker("CD19", "AAAA")])
    with pytest.raises(AssertionError, match="sequence_1"):
        PNAAntibodyPanel.concatenate([left, same_sequence])


def test_hashing_marker_ids_need_a_numeric_suffix_and_must_not_nest():
    missing_suffix = pd.DataFrame(
        [_marker("B2M", "AAAA", sample_hashing=True)]
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="must end with -<digits>"):
        PNAAntibodyPanel(
            missing_suffix,
            AntibodyPanelMetadata(name="base", version="1.0.0"),
        )

    nested = pd.DataFrame(
        [
            _marker("B2M-1", "AAAA", sample_hashing=True),
            _marker("B2M-1-1", "CCCC", sample_hashing=True),
        ]
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="must not collapse"):
        PNAAntibodyPanel(
            nested,
            AntibodyPanelMetadata(name="base", version="1.0.0"),
        )


def test_panel_tables_roundtrip_and_legacy_uns(tmp_path: Path):
    panel = _panel(
        "base",
        "1.0.0",
        [_marker("CD3", "AAAA"), _marker("CD19", "CCCC", control=True)],
    )
    path = tmp_path / "sample.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_metadata({"sample_name": "sample"})
        writer.write_panel(panel)
        # A minimal AnnData so the file is still a pxl for readers that only
        # need the panel tables.
        connection = writer.get_connection()
        connection.execute("CREATE TABLE edgelist (umi1 INTEGER)")
        connection.execute('CREATE TABLE "__adata__X" (index VARCHAR)')
        connection.execute('CREATE TABLE "__adata__obs" (index VARCHAR)')
        connection.execute('CREATE TABLE "__adata__var" (index VARCHAR)')

    loaded = PxlFile(path).read_panel()
    assert loaded is not None
    assert loaded.name == "base"
    assert loaded.version == "1.0.0"
    assert loaded.sources[0].metadata.product == "proxiome"
    assert_frame_equal(loaded.df, panel.df)

    legacy = tmp_path / "legacy.pxl"
    adata = AnnData(X=np.zeros((1, panel.df.shape[0])), var=panel.df.copy())
    adata.uns["panel_metadata"] = panel.metadata.model_dump()
    adata.uns["panel_metadata"]["panel_columns"] = list(panel.df.columns)
    with PixelFileWriter(legacy) as writer:
        writer.write_metadata({"sample_name": "legacy"})
        writer.write_adata(adata)
    legacy_panel = PxlFile(legacy).read_panel()
    assert legacy_panel is not None
    assert legacy_panel.name == "base"
    assert_frame_equal(legacy_panel.df, panel.df)


def test_patch_bump_is_per_source_and_skips_collapsed_hashing_clones():
    old_base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")], product="kit")
    new_base = _panel("base", "1.0.1", [_marker("CD3E", "AAAA")], product="kit")
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")], product="kit")
    old = PNAAntibodyPanel.concatenate([old_base, addon])
    new = PNAAntibodyPanel.concatenate([new_base, addon])

    old_adata = AnnData(var=pd.DataFrame(index=["CD3", "CD19"]))
    new_adata = AnnData(var=pd.DataFrame(index=["CD3E", "CD19"]))
    upgraded, renames, _hash_renames = align_panel_patches(
        [old, new], [old_adata, new_adata]
    )

    assert upgraded[0].sources[0].metadata.version == "1.0.1"
    assert upgraded[0].sources[1].metadata.version == "2.0.0"
    assert "CD3E" in upgraded[0].markers
    assert renames[0] == {"CD3": "CD3E"}
    assert renames[1] == {}

    hashing = _panel(
        "hash",
        "1.0.0",
        [
            _marker("B2M-1", "GGGG", sample_hashing=True),
            _marker("CD3", "AAAA"),
        ],
        product="kit",
    )
    hashing_next = _panel(
        "hash",
        "1.0.1",
        [
            _marker("B2M-1", "GGGG", sample_hashing=True),
            _marker("CD3E", "AAAA"),
        ],
        product="kit",
    )
    collapsed = AnnData(var=pd.DataFrame(index=["B2M", "CD3"]))
    current = AnnData(var=pd.DataFrame(index=["B2M-1", "CD3E"]))
    with pytest.raises(ValueError, match="Missing markers"):
        align_panel_patches(
            [hashing, hashing_next],
            [collapsed, current],
            pxl_file_metadata=[{"hashing_collapsed": False}, {}],
        )
    upgraded, renames, _hash_renames = align_panel_patches(
        [hashing, hashing_next],
        [collapsed, current],
        pxl_file_metadata=[{"hashing_collapsed": True}, {}],
    )
    assert renames[0]["CD3"] == "CD3E"
    assert "B2M-1" in upgraded[0].markers


def test_pxl_fixture_roundtrip(pxl_file):
    panel = read_dataset_panel(read(pxl_file))
    assert panel.name == "test-pna-panel"
    assert "panel_metadata" not in read(pxl_file).adata().uns
