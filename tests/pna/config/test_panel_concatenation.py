"""Tests for concatenating several antibody panels into one.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import (
    PNAAntibodyPanel,
    align_panel_patches,
    aligned_dataset_panel,
)
from pixelator.pna.pixeldataset import read
from pixelator.pna.pixeldataset.io import PixelFileWriter, read_dataset_panel


def _panel(
    name: str, version: str, rows: list[dict], product: str = "proxiome"
) -> PNAAntibodyPanel:
    frame = pd.DataFrame(rows).set_index("marker_id")
    return PNAAntibodyPanel.from_metadata(
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
    right = PNAAntibodyPanel.from_metadata(
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
        sources,
        pd.Series({"CD3": 0, "CD19": 1}, dtype="int64"),
    )
    swapped = PNAAntibodyPanel(
        frame,
        sources,
        pd.Series({"CD3": 1, "CD19": 0}, dtype="int64"),
    )
    assert assigned != swapped

    reordered = PNAAntibodyPanel.concatenate([addon, base])
    assert PNAAntibodyPanel.concatenate([base, addon]) == reordered


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


def test_source_as_panel_splits_a_concatenation():
    base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")])
    combined = PNAAntibodyPanel.concatenate([base, addon])

    assert combined.source_as_panel(0) == base
    assert combined.source_as_panel(1) == addon
    assert combined.copy() == combined

    with pytest.raises(ValueError, match="zero or greater"):
        combined.source_as_panel(-1)
    with pytest.raises(ValueError, match="zero or greater"):
        combined.replace_source(-1, base)


def test_replace_source_blanks_optional_columns_missing_from_one_side():
    old_base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")], product="kit")
    new_base = _panel(
        "base",
        "1.0.1",
        [_marker("CD3E", "AAAA", uniprot_id="P07766")],
        product="kit",
    )
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")], product="kit")
    old = PNAAntibodyPanel.concatenate([old_base, addon])
    new = PNAAntibodyPanel.concatenate([new_base, addon])

    replaced = old.replace_source(0, new_base)

    assert replaced.df.loc["CD3E", "uniprot_id"] == "P07766"
    assert replaced.df.loc["CD19", "uniprot_id"] == ""
    assert replaced == new
    assert aligned_dataset_panel([old, new]) == new

    addon_with_uniprot = _panel(
        "addon",
        "2.0.0",
        [_marker("CD19", "CCCC", uniprot_id="P15391")],
        product="kit",
    )
    newer_without = _panel("base", "1.0.1", [_marker("CD3E", "AAAA")], product="kit")
    kept_column = PNAAntibodyPanel.concatenate([old_base, addon_with_uniprot])
    replaced_other_side = kept_column.replace_source(0, newer_without)
    assert replaced_other_side.df.loc["CD3E", "uniprot_id"] == ""
    assert replaced_other_side.df.loc["CD19", "uniprot_id"] == "P15391"


def test_concatenate_rejects_duplicate_marker_and_sequence():
    left = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    same_marker = _panel("addon", "1.0.0", [_marker("CD3", "CCCC")])
    with pytest.raises(AssertionError, match="marker_id were not unique"):
        PNAAntibodyPanel.concatenate([left, same_marker])

    same_sequence = _panel("addon", "1.0.0", [_marker("CD19", "AAAA")])
    with pytest.raises(AssertionError, match="sequence_1"):
        PNAAntibodyPanel.concatenate([left, same_sequence])


def test_source_as_panel_drops_columns_added_by_another_source():
    with_uniprot = _panel(
        "base",
        "1.0.0",
        [_marker("CD3", "AAAA", uniprot_id="P01730")],
    )
    blank_uniprot = _panel(
        "blank",
        "1.0.0",
        [_marker("CD19", "CCCC", uniprot_id="")],
    )
    no_uniprot = _panel("addon", "1.0.0", [_marker("CD4", "TTTT")])
    hashing_flag = _panel(
        "hash",
        "1.0.0",
        [_marker("ACTB", "GGGG", sample_hashing=False)],
    )
    combined = PNAAntibodyPanel.concatenate(
        [with_uniprot, blank_uniprot, no_uniprot, hashing_flag]
    )

    assert list(combined.source_as_panel(0).df.columns) == list(with_uniprot.df.columns)
    assert combined.df.loc["CD4", "uniprot_id"] == ""
    assert combined.source_as_panel(1).df.loc["CD19", "uniprot_id"] == ""
    assert "uniprot_id" not in combined.source_as_panel(2).df.columns
    assert "sample_hashing" not in combined.source_as_panel(2).df.columns
    assert list(combined.source_as_panel(3).df.columns) == list(hashing_flag.df.columns)


def test_read_dataset_panel_rejects_a_file_from_before_0_22(tmp_path: Path):
    path = tmp_path / "old.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_metadata({"sample_name": "old", "panel_name": "base"})
        connection = writer.get_connection()
        connection.execute("CREATE TABLE edgelist (umi1 INTEGER)")
        connection.execute('CREATE TABLE "__adata__X" (index VARCHAR)')
        connection.execute('CREATE TABLE "__adata__obs" (index VARCHAR)')
        connection.execute('CREATE TABLE "__adata__var" (index VARCHAR)')
        connection.execute('CREATE TABLE "__adata__uns" (value JSON)')

    with pytest.raises(ValueError, match="current version of the software"):
        read_dataset_panel(read(path))


def test_aligned_dataset_panel_ignores_source_order():
    base_old = _panel("base", "1.0.0", [_marker("CD3", "AAAA")], product="kit")
    base_new = _panel("base", "1.0.1", [_marker("CD3E", "AAAA")], product="kit")
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")], product="kit")
    forward = PNAAntibodyPanel.concatenate([base_old, addon])
    reverse = PNAAntibodyPanel.concatenate([addon, base_new])

    assert aligned_dataset_panel([forward, reverse]) == PNAAntibodyPanel.concatenate(
        [base_new, addon]
    )


def test_patch_bump_matches_when_the_added_column_is_not_last():
    old_base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")], product="kit")
    new_frame = pd.DataFrame([_marker("CD3E", "AAAA")]).set_index("marker_id")
    new_frame["uniprot_id"] = "P07766"
    new_frame = new_frame[["control", "uniprot_id", "sequence_1", "sequence_2"]]
    new_base = PNAAntibodyPanel.from_metadata(
        new_frame,
        AntibodyPanelMetadata(name="base", version="1.0.1", product="kit"),
        file_name="base.csv",
    )
    addon = _panel("addon", "2.0.0", [_marker("CD19", "CCCC")], product="kit")
    old = PNAAntibodyPanel.concatenate([old_base, addon])
    new = PNAAntibodyPanel.concatenate([new_base, addon])
    updated, _, _ = align_panel_patches([old, new])

    assert list(updated[0].df.columns) != list(updated[1].df.columns)
    assert updated[0] == updated[1]
    assert aligned_dataset_panel([old, new]) == new


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
