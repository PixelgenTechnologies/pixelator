"""Tests for aligning panels that differ in patch version.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging

import pandas as pd
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import (
    PanelConflictError,
    PNAAntibodyPanel,
    align_panels,
)


def _marker(marker_id: str, sequence: str, **extra) -> dict:
    return {
        "marker_id": marker_id,
        "control": False,
        "sequence_1": sequence,
        "sequence_2": sequence,
        **extra,
    }


def _hashing_marker(marker_id: str, sequence: str) -> dict:
    return _marker(marker_id, sequence, sample_hashing="yes")


def _panel(
    version: str, rows: list[dict], name: str = "panel", product: str = "kit"
) -> PNAAntibodyPanel:
    return PNAAntibodyPanel.from_metadata(
        pd.DataFrame(rows).set_index("marker_id"),
        AntibodyPanelMetadata(name=name, version=version, product=product),
    )


def test_alignment_maps_each_sequence_pair_to_the_newest_row():
    old = _panel("1.0.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("1.0.1", [_marker("CD3E", "AAAA"), _marker("CD4", "CCCC")])

    alignment = align_panels([old, new])

    assert alignment.rows[("AAAA", "AAAA")]["marker_id"] == "CD3E"
    assert alignment.rows[("CCCC", "CCCC")]["marker_id"] == "CD4"


def test_newest_row_wins_whatever_the_input_order():
    old = _panel("1.0.0", [_marker("CD3", "AAAA", uniprot_id="P01730")])
    new = _panel("1.0.10", [_marker("CD3E", "AAAA", uniprot_id="P01732")])

    for panels in ([old, new], [new, old]):
        row = align_panels(panels).rows[("AAAA", "AAAA")]
        assert row["marker_id"] == "CD3E"
        assert row["uniprot_id"] == "P01732"


def test_marker_renames_only_lists_ids_that_change():
    old = _panel("1.0.0", [_marker("CD3", "AAAA"), _marker("CD4", "CCCC")])
    new = _panel("1.0.1", [_marker("CD3E", "AAAA"), _marker("CD4", "CCCC")])

    alignment = align_panels([old, new])

    assert alignment.marker_renames(old) == {"CD3": "CD3E"}
    assert alignment.marker_renames(new) == {}


def test_identical_panels_have_no_renames():
    panel = _panel("1.0.0", [_marker("CD3", "AAAA")])

    alignment = align_panels([panel, panel])

    assert alignment.marker_renames(panel) == {}


def test_a_single_panel_aligns_with_itself():
    panel = _panel("1.0.0", [_marker("CD3", "AAAA")])

    assert align_panels([panel]).marker_renames(panel) == {}


def test_each_source_of_a_concatenated_panel_is_aligned_on_its_own():
    base_old = _panel("1.0.0", [_marker("CD3", "AAAA")], name="base", product="base")
    addon_old = _panel(
        "2.0.0", [_marker("CD19", "CCCC")], name="addon", product="addon"
    )
    base_new = _panel("1.0.1", [_marker("CD3E", "AAAA")], name="base", product="base")
    addon_new = _panel(
        "2.0.1", [_marker("CD19A", "CCCC")], name="addon", product="addon"
    )
    old = PNAAntibodyPanel.concatenate([base_old, addon_old])
    new = PNAAntibodyPanel.concatenate([addon_new, base_new])

    alignment = align_panels([old, new])

    assert alignment.marker_renames(old) == {"CD3": "CD3E", "CD19": "CD19A"}
    assert alignment.marker_renames(new) == {}


@pytest.mark.parametrize("version", ["1.1.0", "2.0.0"])
def test_sources_of_the_same_product_with_different_major_or_minor_are_rejected(
    version,
):
    old = _panel("1.0.0", [_marker("CD3", "AAAA")])
    other = _panel(version, [_marker("CD3", "AAAA")])

    with pytest.raises(ValueError, match="same major and minor version"):
        align_panels([old, other])


def test_a_product_with_two_major_minor_versions_among_addons_is_rejected():
    base = _panel("1.0.0", [_marker("CD3", "AAAA")], name="base", product="base")
    addon_a = _panel("1.0.0", [_marker("CD19", "CCCC")], name="addon", product="addon")
    addon_b = _panel("1.1.0", [_marker("CD20", "GGGG")], name="addon", product="addon")

    with pytest.raises(ValueError, match="addon: 1.0, 1.1"):
        align_panels(
            [
                PNAAntibodyPanel.concatenate([base, addon_a]),
                PNAAntibodyPanel.concatenate([base, addon_b]),
            ]
        )


def test_panels_with_different_addons_align_into_base_plus_both_addons():
    base_old = _panel("1.0.0", [_marker("CD3", "AAAA")], name="base", product="base")
    base_new = _panel("1.0.1", [_marker("CD3E", "AAAA")], name="base", product="base")
    addon_a = _panel("1.0.0", [_marker("CD19", "CCCC")], name="a", product="addon_a")
    addon_b = _panel("1.0.0", [_marker("CD20", "GGGG")], name="b", product="addon_b")
    sample_a = PNAAntibodyPanel.concatenate([base_old, addon_a])
    sample_b = PNAAntibodyPanel.concatenate([base_new, addon_b])

    alignment = align_panels([sample_a, sample_b])

    assert alignment.marker_renames(sample_a) == {"CD3": "CD3E"}
    assert alignment.marker_renames(sample_b) == {}
    # the markers of both add-ons are kept
    assert alignment.rows[("CCCC", "CCCC")]["marker_id"] == "CD19"
    assert alignment.rows[("GGGG", "GGGG")]["marker_id"] == "CD20"


def test_sources_without_a_product_are_matched_on_their_name():
    old = _panel("1.0.0", [_marker("CD3", "AAAA")], name="custom", product=None)
    new = _panel("1.0.1", [_marker("CD3E", "AAAA")], name="custom", product=None)

    alignment = align_panels([old, new])

    assert alignment.marker_renames(old) == {"CD3": "CD3E"}


def test_no_panels_is_rejected():
    with pytest.raises(ValueError, match="At least one panel"):
        align_panels([])


def test_hashing_clones_and_their_collapsed_id_are_renamed():
    old = _panel(
        "1.0.0",
        [_hashing_marker("B2M-1", "AAAA"), _hashing_marker("B2M-2", "CCCC")],
    )
    new = _panel(
        "1.0.1",
        [_hashing_marker("B3M-1", "AAAA"), _hashing_marker("B3M-2", "CCCC")],
    )

    renames = align_panels([old, new]).marker_renames(old)

    assert renames == {
        "B2M-1": "B3M-1",
        "B2M-2": "B3M-2",
        "B2M": "B3M",
    }


def test_collapsed_rename_is_left_out_when_clones_disagree_on_the_base():
    old = _panel(
        "1.0.0",
        [_hashing_marker("B2M-1", "AAAA"), _hashing_marker("B2M-2", "CCCC")],
    )
    new = _panel(
        "1.0.1",
        [_hashing_marker("B3M-1", "AAAA"), _hashing_marker("B4M-2", "CCCC")],
    )

    renames = align_panels([old, new]).marker_renames(old)

    assert renames == {"B2M-1": "B3M-1", "B2M-2": "B4M-2"}


def test_collapsed_rename_is_left_out_when_the_base_is_a_panel_marker():
    old = _panel(
        "1.0.0",
        [_marker("B2M", "GGGG"), _hashing_marker("B2M-1", "AAAA")],
    )
    new = _panel(
        "1.0.1",
        [_marker("B2M", "GGGG"), _hashing_marker("B3M-1", "AAAA")],
    )

    renames = align_panels([old, new]).marker_renames(old)

    assert renames == {"B2M-1": "B3M-1"}


def test_unchanged_hashing_markers_have_no_renames():
    panel = _panel("1.0.0", [_hashing_marker("B2M-1", "AAAA")])

    assert align_panels([panel, panel]).marker_renames(panel) == {}


def _two_products(first_rows: list[dict], second_rows: list[dict], *versions: str):
    first_version, second_version = versions
    return (
        _panel(first_version, first_rows, name="a", product="a"),
        _panel(second_version, second_rows, name="b", product="b"),
    )


def test_one_marker_id_with_two_sequence_pairs_is_rejected():
    first, second = _two_products(
        [_marker("B2M-1", "AAAA")], [_marker("B2M-1", "CCCC")], "1.0.0", "1.0.0"
    )

    with pytest.raises(PanelConflictError, match="marker id 'B2M-1'") as err:
        align_panels([first, second])

    assert "a 1.0.0" in str(err.value)
    assert "b 1.0.0" in str(err.value)
    assert "force=True" in str(err.value)


def test_an_id_conflict_within_one_product_is_rejected():
    old = _panel("1.0.0", [_marker("CD3", "AAAA")])
    new = _panel("1.0.1", [_marker("CD3", "CCCC")])

    with pytest.raises(PanelConflictError, match="marker id 'CD3'"):
        align_panels([old, new])


def test_the_same_pair_under_two_ids_in_different_products_is_rejected():
    first, second = _two_products(
        [_marker("CD3", "AAAA")], [_marker("CD3E", "AAAA")], "1.0.0", "2.0.0"
    )

    with pytest.raises(PanelConflictError, match="have different marker ids"):
        align_panels([first, second])


def test_the_same_pair_and_id_in_different_products_is_not_a_conflict():
    first, second = _two_products(
        [_marker("CD3", "AAAA")], [_marker("CD3", "AAAA")], "1.0.0", "1.0.0"
    )

    assert align_panels([first, second]).marker_renames(first) == {}


def test_a_rename_within_a_product_is_not_a_conflict():
    old = _panel("1.0.0", [_marker("CD3", "AAAA")])
    new = _panel("1.0.1", [_marker("CD3E", "AAAA")])

    alignment = align_panels([new, old])

    assert alignment.marker_renames(old) == {"CD3": "CD3E"}


def test_only_the_first_conflicts_are_listed():
    first, second = _two_products(
        [_marker(f"M{i}", f"AA{'ACGT'[i // 4]}{'ACGT'[i % 4]}") for i in range(12)],
        [_marker(f"M{i}", f"CC{'ACGT'[i // 4]}{'ACGT'[i % 4]}") for i in range(12)],
        "1.0.0",
        "1.0.0",
    )

    with pytest.raises(PanelConflictError, match="and 2 more"):
        align_panels([first, second])


def test_force_warns_about_conflicts_instead_of_raising(caplog):
    first, second = _two_products(
        [_marker("B2M-1", "AAAA"), _marker("CD3", "GGGG")],
        [_marker("B2M-1", "CCCC"), _marker("CD3E", "GGGG")],
        "1.0.0",
        "1.0.0",
    )

    with caplog.at_level(logging.WARNING):
        alignment = align_panels([first, second], force=True)

    assert "marker id 'B2M-1'" in caplog.text
    assert "have different marker ids" in caplog.text
    assert set(alignment.rows) == {
        ("AAAA", "AAAA"),
        ("CCCC", "CCCC"),
        ("GGGG", "GGGG"),
    }


def test_force_keeps_the_last_product_for_a_pair_shared_by_two_products():
    first, second = _two_products(
        [_marker("CD3", "AAAA")], [_marker("CD3E", "AAAA")], "2.0.0", "1.0.0"
    )

    for panels, expected in (([first, second], "CD3E"), ([second, first], "CD3")):
        row = align_panels(panels, force=True).rows[("AAAA", "AAAA")]
        assert row["marker_id"] == expected


def test_force_keeps_the_newest_row_of_an_id_with_several_pairs():
    v0 = _panel("1.0.0", [_marker("CD3", "AAAA")])
    v1 = _panel("1.0.1", [_marker("CD3", "CCCC")])
    v2 = _panel("1.0.2", [_marker("CD3", "AAAA")])

    alignment = align_panels([v0, v1, v2], force=True)

    # The order of the rows follows their recency, so the last row is newest.
    assert list(alignment.rows) == [("CCCC", "CCCC"), ("AAAA", "AAAA")]
