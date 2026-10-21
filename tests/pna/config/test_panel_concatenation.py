"""Tests for concatenating several antibody panels into one.

Copyright © 2026 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import (
    PNAAntibodyPanel,
    load_antibody_panel,
    load_antibody_panels,
)


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


def _write_panel_csv(path, name: str, rows: list[tuple[str, str]]):
    header = f"# ---\n# name: {name}\n# product: kit\n# version: 1.0.0\n# ---\n"
    lines = ["marker_id,control,sequence_1,sequence_2"]
    lines += [f"{marker},,{sequence},{sequence}" for marker, sequence in rows]
    path.write_text(header + "\n".join(lines) + "\n")
    return path


def test_load_antibody_panels_accepts_one_name_or_a_sequence():
    name = "proxiome-v1-immuno-155-v1.0"
    single = load_antibody_panel(pna_config, name)

    assert load_antibody_panels(pna_config, name) == single
    assert load_antibody_panels(pna_config, [name]) == single


def test_load_antibody_panels_concatenates_panel_files(tmp_path):
    left = _write_panel_csv(tmp_path / "left.csv", "left", [("CD3", "AAAA")])
    right = _write_panel_csv(tmp_path / "right.csv", "right", [("CD19", "CCCC")])

    panel = load_antibody_panels(pna_config, [left, right])

    assert [source.metadata.name for source in panel.sources] == ["left", "right"]
    assert panel.markers == ["CD3", "CD19"]


def test_load_antibody_panels_rejects_duplicate_markers(tmp_path):
    left = _write_panel_csv(tmp_path / "left.csv", "left", [("CD3", "AAAA")])
    right = _write_panel_csv(tmp_path / "right.csv", "right", [("CD3", "CCCC")])

    with pytest.raises(ValueError, match="marker_id 'CD3'"):
        load_antibody_panels(pna_config, [left, right])


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
        [_marker("B2M-1", "GGGG", sample_hashing="yes")],
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
    with pytest.raises(ValueError, match="marker_id 'CD3' is in both"):
        PNAAntibodyPanel.concatenate([left, same_marker])

    same_sequence = _panel("addon", "1.0.0", [_marker("CD19", "AAAA")])
    with pytest.raises(ValueError, match="sequence_1 'AAAA' is in both"):
        PNAAntibodyPanel.concatenate([left, same_sequence])


def test_concatenate_names_the_sources_that_overlap():
    base = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])
    addon = _panel("addon", "2.1.0", [_marker("CD19", "CCCC")])
    other = _panel("other", "3.0.0", [_marker("CD3", "GGGG")])

    with pytest.raises(ValueError) as error:
        PNAAntibodyPanel.concatenate([base, addon, other])

    message = str(error.value)
    assert "base 1.0.0 (input 1)" in message
    assert "other 3.0.0 (input 3)" in message
    assert "addon" not in message


def test_concatenate_rejects_the_same_panel_twice():
    panel = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])

    with pytest.raises(ValueError, match="input 1.*input 2"):
        PNAAntibodyPanel.concatenate([panel, panel])


def test_concatenate_reports_overlaps_in_every_column_and_caps_the_list():
    rows = [_marker(f"M{i}", f"{'ACGT'[i // 4]}{'ACGT'[i % 4]}CC") for i in range(16)]
    left = _panel("base", "1.0.0", rows)
    right = _panel("addon", "1.0.0", rows)

    with pytest.raises(ValueError) as error:
        PNAAntibodyPanel.concatenate([left, right])

    lines = str(error.value).splitlines()
    # a header, ten overlaps, and a count of the rest
    assert len(lines) == 12
    assert lines[-1] == "and 38 more"


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
        [_marker("ACTB", "GGGG", sample_hashing="")],
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


def test_concatenate_blanks_unknown_column_missing_from_one_source():
    flagged = _panel("base", "1.0.0", [_marker("CD3", "AAAA", nuclear="yes")])
    plain = _panel("addon", "1.0.0", [_marker("CD19", "CCCC")])

    combined = PNAAntibodyPanel.concatenate([flagged, plain])

    assert combined.df.loc["CD3", "nuclear"] == "yes"
    assert combined.df.loc["CD19", "nuclear"] == ""
    assert "nuclear" not in combined.source_as_panel(1).df.columns


def test_panel_requires_a_source():
    panel = _panel("base", "1.0.0", [_marker("CD3", "AAAA")])

    with pytest.raises(ValueError, match="at least one source"):
        PNAAntibodyPanel(panel.df, [], panel.marker_source_ids)
