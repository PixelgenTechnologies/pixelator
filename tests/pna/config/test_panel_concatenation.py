"""Tests for concatenating several antibody panels into one.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel


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
