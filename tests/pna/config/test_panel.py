"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import polars as pl
import pytest
import ruamel.yaml as yaml
from anndata import AnnData
from pandas.testing import assert_frame_equal

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.anndata import add_panel_information
from pixelator.pna.config.panel import (
    PanelType,
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNAAntibodyPanelCombination,
    PNAAntibodyPanelDiff,
    PNABasePanel,
    PNASampleHashingPanel,
    collapsed_hashing_marker_id,
    sample_hashing_mask,
    split_hashing_marker_id,
)
from pixelator.pna.pixeldataset import read
from pixelator.pna.utils.sample_calling_uns import (
    sample_calling_hashing_collapsed,
    set_sample_calling_collapsed,
)


@pytest.fixture
def panel_df():
    """Panel df."""
    data = {
        "marker_id": ["marker1", "marker2", "marker3"],
        "uniprot_id": ["P61769", "P05107", "P15391"],
        "control": [False, True, False],
        "nuclear": [True, False, True],
        "sequence_1": ["ATCG", "GCTA", "ATCC"],
        "sequence_2": ["ATCG", "GCTA", "ATCC"],
    }
    return pd.DataFrame(data).set_index("marker_id")


def _apply_edgelist_marker_id_mapping(
    diff: PNAAntibodyPanelDiff,
    edgelist: pl.DataFrame,
    *,
    collapsed: bool = False,
) -> pl.DataFrame:
    """Apply ``diff.edgelist_marker_id_mapping`` to a standalone edgelist."""
    mapping = diff.edgelist_marker_id_mapping(collapsed=collapsed)
    if not mapping:
        return edgelist
    map_df = pl.DataFrame(
        {"old_id": list(mapping.keys()), "new_id": list(mapping.values())}
    )
    upgraded = edgelist
    for column in ("marker_1", "marker_2"):
        if column not in upgraded.columns:
            continue
        upgraded = (
            upgraded.join(map_df, left_on=column, right_on="old_id", how="left")
            .with_columns(pl.coalesce("new_id", column).alias(column))
            .drop("new_id")
        )
    return upgraded


def test_panel_validation(panel_df):
    # all is ok
    """Verify panel validation.

    Args:
        panel_df: panel df.
    """
    metadata = {
        "name": "test_panel",
        "version": "0.0.0",
        "description": "panel description",
        "aliases": ["test_alias"],
    }
    panel = PNAAntibodyPanelCombination(
        PartialPNAAntibodyPanel(
            panel_df,
            AntibodyPanelMetadata(**metadata),
            file_name="test.csv",
        )
    )

    assert panel.name == metadata["name"]
    assert panel.version == metadata["version"]
    assert panel.description == metadata["description"]
    assert panel.aliases == metadata["aliases"]

    assert panel.markers_control == ["marker2"]
    assert panel.markers == ["marker1", "marker2", "marker3"]
    assert_frame_equal(
        panel.df.drop(columns=["partial_panel_name", "partial_panel_type"]), panel_df
    )
    assert panel.filename == "test.csv"
    assert panel.filepath is None
    assert panel.size == 3


def test_combination_marker_helpers_follow_df_after_add_panel(panel_df, hashing_panel):
    """Marker helpers must stay in sync with df after mutating membership."""
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel",
            version="0.0.0",
            panel_type=PanelType.BASE,
        ),
    )
    combo = PNAAntibodyPanelCombination(base)

    # Prime any former cache before membership changes.
    assert combo.markers == ["marker1", "marker2", "marker3"]
    assert combo.markers_control == ["marker2"]
    assert combo.size == 3

    combo.add_hashing_panel(hashing_panel)
    assert combo.markers == list(combo.df.index.unique())
    assert combo.markers_control == list(combo.df[combo.df["control"]].index)
    assert combo.size == combo.df.shape[0]
    assert "HM-1" in combo.markers
    assert combo.size == 5

    addon_df = pd.DataFrame(
        {
            "marker_id": ["addon1"],
            "uniprot_id": ["P12345"],
            "control": [True],
            "nuclear": [False],
            "sequence_1": ["AAAA"],
            "sequence_2": ["TTTT"],
        }
    ).set_index("marker_id")
    addon = PNAAddonPanel(
        addon_df,
        AntibodyPanelMetadata(
            name="addon-panel",
            version="0.0.0",
            panel_type=PanelType.ADDON,
        ),
    )
    combo.add_panel(addon)
    assert combo.markers == list(combo.df.index.unique())
    assert combo.markers_control == list(combo.df[combo.df["control"]].index)
    assert combo.size == combo.df.shape[0]
    assert combo.markers_control == ["marker2", "addon1"]
    assert combo.size == 6


def test_sample_hashing_mask_accepts_bool_string_and_float_upcast():
    """Hashing flags must survive concat upcasts and legacy string encodings."""
    bool_col = pd.Series([True, False, pd.NA], dtype="boolean")
    assert sample_hashing_mask(bool_col).tolist() == [True, False, False]

    yes_no = pd.Series(["yes", "no", "YES", None, ""])
    assert sample_hashing_mask(yes_no).tolist() == [True, False, True, False, False]

    true_false = pd.Series(["True", "False", "true"])
    assert sample_hashing_mask(true_false).tolist() == [True, False, True]

    float_upcast = pd.Series([1.0, 0.0, float("nan")])
    assert sample_hashing_mask(float_upcast).tolist() == [True, False, False]

    object_float = pd.Series([1.0, float("nan"), True, False])
    assert sample_hashing_mask(object_float).tolist() == [True, False, True, False]


def test_split_hashing_marker_id_strips_hash_group_only():
    """Suffix parsing is for hashing ids already selected via sample_hashing."""
    assert split_hashing_marker_id("B2M-1") == ("B2M", "1")
    assert split_hashing_marker_id("IL-2-8") == ("IL-2", "8")
    assert split_hashing_marker_id("MarkerA") is None
    assert collapsed_hashing_marker_id("B2M-1") == "B2M"
    assert collapsed_hashing_marker_id("MarkerA") == "MarkerA"


def _mixed_hyphen_name_panel(
    *, version: str = "0.1.0", rename: dict[str, str] | None = None
):
    """Panel with non-hashing PD-1/TIM-3 and hashing B2M-1/B2M-2."""
    df = pd.DataFrame(
        {
            "marker_id": ["PD-1", "TIM-3", "B2M-1", "B2M-2"],
            "control": [False, False, False, False],
            "sequence_1": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sequence_2": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sample_hashing": [False, False, True, True],
        }
    ).set_index("marker_id")
    if rename:
        df = df.rename(index=rename)
    return PartialPNAAntibodyPanel(
        df,
        AntibodyPanelMetadata(
            name="mixed-hyphen", version=version, product="mixed-hyphen"
        ),
    )


def test_hyphenated_non_hashing_markers_are_not_hashing_by_name():
    """PD-1 / TIM-3 match the hashing name pattern but only the column counts."""
    panel_old = _mixed_hyphen_name_panel()
    panel_new = _mixed_hyphen_name_panel(
        version="0.1.1",
        rename={"PD-1": "PD-9", "B2M-1": "NEWB2M-1", "B2M-2": "NEWB2M-2"},
    )
    diff = PNAAntibodyPanelDiff(panel_old, panel_new)

    assert panel_old.hashing_marker_ids == {"B2M-1", "B2M-2"}
    assert diff.marker_id_mapping()["PD-1"] == "PD-9"
    assert "PD-1" not in diff.collapsed_hashing_marker_id_mapping()
    assert "TIM-3" not in diff.collapsed_hashing_marker_id_mapping()
    assert diff.collapsed_hashing_marker_id_mapping() == {"B2M": "NEWB2M"}

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["PD-1", "TIM-3", "B2M"], name="marker_id")),
    )
    adata = add_panel_information(adata, PNAAntibodyPanelCombination(panel_old))
    set_sample_calling_collapsed(adata, True)
    upgraded = diff.upgrade_adata(adata)

    assert "PD-9" in upgraded.var.index
    assert "PD-1" not in upgraded.var.index
    assert "TIM-3" in upgraded.var.index
    assert "B2M" not in upgraded.var.index
    assert "NEWB2M" in upgraded.var.index

    missing_pd1 = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["TIM-3", "B2M"], name="marker_id")),
    )
    missing_pd1 = add_panel_information(
        missing_pd1, PNAAntibodyPanelCombination(panel_old)
    )
    set_sample_calling_collapsed(missing_pd1, True)
    with pytest.raises(ValueError, match="missing panel clones"):
        diff.upgrade_adata(missing_pd1)


def test_combination_keeps_hashing_flags_when_base_omits_sample_hashing(
    panel_df, hashing_panel
):
    """Base panels without sample_hashing must not hide hashing markers after concat."""
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel",
            version="0.0.0",
            panel_type=PanelType.BASE,
        ),
    )
    assert "sample_hashing" not in base.df.columns
    assert hashing_panel.df["sample_hashing"].dtype == bool

    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    assert combo.name == "base-panel + hash-set-1"
    hashing_col = combo.df["sample_hashing"]
    assert pd.api.types.is_bool_dtype(hashing_col)
    assert hashing_col.loc[["HM-1", "HM-2"]].all()
    assert not hashing_col.loc[panel_df.index].any()
    assert combo.df[sample_hashing_mask(hashing_col)].index.tolist() == ["HM-1", "HM-2"]
    assert combo.hashing_marker_ids == {"HM-1", "HM-2"}


def test_hashing_marker_ids_from_sample_hashing_column_on_base_panel():
    """v2-style combined CSVs flag hashing on the base panel, not a hashing member."""
    combo = PNAAntibodyPanelCombination(_mixed_hyphen_name_panel())
    assert combo.hashing_panels is None
    assert combo.hashing_marker_ids == {"B2M-1", "B2M-2"}


def test_addon_panel_rejects_hashing_markers_with_string_flags(panel_df):
    """Addon validation must treat 'yes' as hashing, not any non-empty string."""
    hashing_addon = panel_df.copy()
    hashing_addon["sample_hashing"] = ["no", "yes", "no"]
    with pytest.raises(AssertionError, match="cannot include hashing markers"):
        PNAAddonPanel(
            hashing_addon,
            AntibodyPanelMetadata(
                name="addon-hash",
                version="0.0.0",
                panel_type=PanelType.ADDON,
            ),
        )

    non_hashing_addon = panel_df.copy()
    non_hashing_addon["sample_hashing"] = "no"
    addon = PNAAddonPanel(
        non_hashing_addon,
        AntibodyPanelMetadata(
            name="addon-ok",
            version="0.0.0",
            panel_type=PanelType.ADDON,
        ),
    )
    assert list(addon.df.index) == list(panel_df.index)


def test_hashing_panel_rejects_mixed_string_sample_hashing_flags(panel_df):
    """A hashing panel must be all hashing after yes/no normalization."""
    mixed = panel_df.copy()
    mixed["sample_hashing"] = ["yes", "no", "yes"]
    with pytest.raises(AssertionError, match="must be 'yes'"):
        PNASampleHashingPanel(
            mixed,
            AntibodyPanelMetadata(
                name="hash-mixed",
                version="0.0.0",
                panel_type=PanelType.SAMPLE_HASHING,
            ),
        )


def test_hashing_panel_requires_hash_group_suffix_on_marker_ids():
    """Hashing marker ids must be NAME-<digits>, e.g. HM-1."""
    df = pd.DataFrame(
        {
            "marker_id": ["HM-1", "HM"],
            "control": [False, False],
            "sequence_1": ["ACTTCCTACC", "GGGCTATGGT"],
            "sequence_2": ["ACTTCCTACC", "GGGCTATGGT"],
            "sample_hashing": [True, True],
        }
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match=r"-<digits>"):
        PNASampleHashingPanel(
            df,
            AntibodyPanelMetadata(
                name="hash-nosuffix",
                version="0.0.0",
                panel_type=PanelType.SAMPLE_HASHING,
            ),
        )


def test_hashing_flagged_rows_on_any_panel_require_hash_group_suffix():
    """Combined v2 CSVs flag hashing on a base/partial panel, not only hashing members."""
    df = pd.DataFrame(
        {
            "marker_id": ["B2M", "B2M-1"],
            "control": [False, False],
            "sequence_1": ["AAAAAA", "CCCCCC"],
            "sequence_2": ["AAAAAA", "CCCCCC"],
            "sample_hashing": [True, True],
        }
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match=r"-<digits>"):
        PartialPNAAntibodyPanel(
            df,
            AntibodyPanelMetadata(name="mixed-hash", version="0.0.0"),
        )


def test_hashing_panel_rejects_nested_hash_group_suffixes():
    """B2M-1-1 is invalid when B2M-1 is also a hashing marker."""
    df = pd.DataFrame(
        {
            "marker_id": ["HM-1", "HM-1-1"],
            "control": [False, False],
            "sequence_1": ["ACTTCCTACC", "GGGCTATGGT"],
            "sequence_2": ["ACTTCCTACC", "GGGCTATGGT"],
            "sample_hashing": [True, True],
        }
    ).set_index("marker_id")
    with pytest.raises(AssertionError, match="collapse to another hashing id"):
        PNASampleHashingPanel(
            df,
            AntibodyPanelMetadata(
                name="hash-nested",
                version="0.0.0",
                panel_type=PanelType.SAMPLE_HASHING,
            ),
        )


def test_combination_rejects_nested_hashing_ids_across_members():
    """Nested hashing ids are invalid even when they sit on different members."""
    base = PNABasePanel(
        pd.DataFrame(
            {
                "marker_id": ["B2M-1"],
                "control": [False],
                "sequence_1": ["AAAAAA"],
                "sequence_2": ["AAAAAA"],
                "sample_hashing": [True],
            }
        ).set_index("marker_id"),
        AntibodyPanelMetadata(
            name="b2m-hash-base",
            version="0.1.0",
            panel_type=PanelType.BASE,
        ),
    )
    hashing = PNASampleHashingPanel(
        pd.DataFrame(
            {
                "marker_id": ["B2M-1-1"],
                "control": [False],
                "sequence_1": ["CCCCCC"],
                "sequence_2": ["CCCCCC"],
                "sample_hashing": [True],
            }
        ).set_index("marker_id"),
        AntibodyPanelMetadata(
            name="b2m-nested-hash",
            version="0.1.0",
            panel_type=PanelType.SAMPLE_HASHING,
        ),
    )
    with pytest.raises(ValueError, match="collapse to another hashing id"):
        PNAAntibodyPanelCombination([base, hashing])


def test_panel_combination_classifies_hashing_panel_regardless_of_order(
    panel_df, hashing_panel
):
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel",
            version="0.0.0",
            panel_type=PanelType.BASE,
        ),
    )
    combo_hashing_first = PNAAntibodyPanelCombination([hashing_panel, base])
    combo_base_first = PNAAntibodyPanelCombination([base, hashing_panel])

    for combo in (combo_hashing_first, combo_base_first):
        assert len(combo.base_panels) == 1
        assert len(combo.hashing_panels) == 1
        assert combo.num_partial_panels == 2


def test_combination_rejects_duplicate_sequences(panel_df):
    meta1 = AntibodyPanelMetadata(
        name="panel-a", version="0.0.0", panel_type=PanelType.BASE
    )
    meta2 = AntibodyPanelMetadata(
        name="panel-b", version="0.0.0", panel_type=PanelType.BASE
    )
    with pytest.raises(ValueError, match="Duplicate sequences found"):
        PNAAntibodyPanelCombination(
            [
                PNABasePanel(panel_df, meta1),
                PNABasePanel(panel_df.copy(), meta2),
            ]
        )


def test_combination_rejects_conflicting_duplicate_marker_id(panel_df):
    meta1 = AntibodyPanelMetadata(
        name="panel-a", version="0.0.0", panel_type=PanelType.BASE
    )
    meta2 = AntibodyPanelMetadata(
        name="panel-b", version="0.0.0", panel_type=PanelType.BASE
    )
    conflicting_df = panel_df.copy()
    conflicting_df.loc["marker1", "sequence_1"] = "TTTT"

    with pytest.raises(ValueError, match="Conflicting duplicate marker_id"):
        PNAAntibodyPanelCombination(
            [
                PNABasePanel(panel_df, meta1),
                PNABasePanel(conflicting_df, meta2),
            ]
        )


def test_add_panel_methods_roll_back_on_conflict(panel_df):
    """Failed adds must not leave a half-updated combination that keeps failing df."""
    combo = PNAAntibodyPanelCombination(
        PNABasePanel(
            panel_df,
            AntibodyPanelMetadata(
                name="panel-a", version="0.0.0", panel_type=PanelType.BASE
            ),
        )
    )

    with pytest.raises(ValueError, match="Duplicate sequences found"):
        combo.add_base_panel(
            PNABasePanel(
                panel_df.copy(),
                AntibodyPanelMetadata(
                    name="panel-b", version="0.0.0", panel_type=PanelType.BASE
                ),
            )
        )
    assert [p.name for p in combo.base_panels] == ["panel-a"]
    assert combo.hashing_panels is None
    assert combo.addon_panels is None
    assert list(combo.df.index) == list(panel_df.index)

    overlapping_seq_df = pd.DataFrame(
        {
            "marker_id": ["extra-1"],
            "uniprot_id": ["P12345"],
            "control": [False],
            "nuclear": [False],
            "sequence_1": [panel_df.loc["marker1", "sequence_1"]],
            "sequence_2": [panel_df.loc["marker1", "sequence_2"]],
        }
    ).set_index("marker_id")

    with pytest.raises(ValueError, match="Duplicate sequences found"):
        combo.add_hashing_panel(
            PNASampleHashingPanel(
                overlapping_seq_df.assign(sample_hashing=True),
                AntibodyPanelMetadata(
                    name="hash-bad",
                    version="0.0.0",
                    panel_type=PanelType.SAMPLE_HASHING,
                ),
            )
        )
    assert combo.hashing_panels is None
    assert list(combo.df.index) == list(panel_df.index)

    with pytest.raises(ValueError, match="Duplicate sequences found"):
        combo.add_addon_panel(
            PNAAddonPanel(
                overlapping_seq_df,
                AntibodyPanelMetadata(
                    name="addon-bad", version="0.0.0", panel_type=PanelType.ADDON
                ),
            )
        )
    assert combo.addon_panels is None
    assert list(combo.df.index) == list(panel_df.index)

    conflicting_df = panel_df.copy()
    conflicting_df.loc["marker1", "sequence_1"] = "TTTT"
    with pytest.raises(ValueError, match="Conflicting duplicate marker_id"):
        combo.add_base_panel(
            PNABasePanel(
                conflicting_df,
                AntibodyPanelMetadata(
                    name="panel-c", version="0.0.0", panel_type=PanelType.BASE
                ),
            )
        )
    assert [p.name for p in combo.base_panels] == ["panel-a"]
    assert list(combo.df.index) == list(panel_df.index)


def test_combination_description_skips_none_and_returns_none_when_empty(
    panel_df, hashing_panel
):
    """Joined description must omit None members rather than the literal 'None'."""
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel",
            version="0.0.0",
            description="Test panel",
            panel_type=PanelType.BASE,
        ),
    )
    mixed = PNAAntibodyPanelCombination([base, hashing_panel])
    assert hashing_panel.metadata.description is None
    assert mixed.description == "Test panel"

    addon_df = pd.DataFrame(
        {
            "marker_id": ["addon1"],
            "uniprot_id": ["P12345"],
            "control": [False],
            "nuclear": [False],
            "sequence_1": ["AAAA"],
            "sequence_2": ["TTTT"],
        }
    ).set_index("marker_id")
    addon = PNAAddonPanel(
        addon_df,
        AntibodyPanelMetadata(
            name="addon-panel",
            version="0.0.0",
            description="Addon panel",
            panel_type=PanelType.ADDON,
        ),
    )
    both = PNAAntibodyPanelCombination([base, addon])
    assert both.description == "Test panel + Addon panel"

    none_combo = PNAAntibodyPanelCombination(
        PNABasePanel(
            panel_df,
            AntibodyPanelMetadata(
                name="base-panel",
                version="0.0.0",
                panel_type=PanelType.BASE,
            ),
        )
    )
    assert none_combo.description is None


def test_combination_aliases_raises_for_multi_panel(panel_df, hashing_panel):
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel", version="0.0.0", panel_type=PanelType.BASE
        ),
    )
    combo = PNAAntibodyPanelCombination([base, hashing_panel])

    with pytest.raises(AttributeError, match="Cannot get aliases"):
        _ = combo.aliases


def test_panel_validation_fails_on_underscores_in_marker_names(panel_df):
    """Verify panel validation fails on underscores in marker names.

    Args:
        panel_df: panel df.
    """
    panel_df.rename(index={"marker1": "marker_1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain underscores.*Offending values:.*",
    ):
        PNAAntibodyPanelCombination(
            PartialPNAAntibodyPanel(
                panel_df,
                AntibodyPanelMetadata(
                    name="mock-name",
                    version="0.0.0",
                ),
            )
        )


def test_panel_validation_fails_on_white_space_in_marker_names(panel_df):
    """Verify panel validation fails on white space in marker names.

    Args:
        panel_df: panel df.
    """
    panel_df.rename(index={"marker1": "marker 1"}, inplace=True)

    with pytest.raises(
        AssertionError,
        match=r".*The marker_id column should not contain white-spaces.*Offending values:.*",
    ):
        PNAAntibodyPanelCombination(
            PartialPNAAntibodyPanel(
                panel_df,
                AntibodyPanelMetadata(
                    name="mock-name",
                    version="0.0.0",
                ),
            )
        )


def test_panel_validation_fails_on_invalid_uniprot_ids(panel_df):
    """Verify panel validation fails on invalid uniprot ids.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = "PAAAAA"

    with pytest.raises(
        AssertionError,
        match=r".*Invalid UniProt IDs found.*Please conform to the naming convention or remove the following IDs:.*",
    ):
        PNAAntibodyPanelCombination(
            PartialPNAAntibodyPanel(
                panel_df,
                AntibodyPanelMetadata(
                    name="mock-name",
                    version="0.0.0",
                ),
            )
        )


def test_panel_validation_ok_on_concatenated_uniprot_ids(panel_df):
    """Verify panel validation ok on concatenated uniprot ids.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = "P05107;P15391"
    PNAAntibodyPanelCombination(
        PartialPNAAntibodyPanel(
            panel_df,
            AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )
    )


def test_panel_validation_ok_uniprotid_empty(panel_df):
    """Verify panel validation ok uniprotid empty.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = ""
    PNAAntibodyPanelCombination(
        PartialPNAAntibodyPanel(
            panel_df,
            AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )
    )


def test_panel_metadata_panel_type_must_match_class(panel_df):
    with pytest.raises(ValueError, match="does not match"):
        PNABasePanel(
            panel_df,
            AntibodyPanelMetadata(
                name="wrong-type",
                version="0.0.0",
                panel_type=PanelType.SAMPLE_HASHING,
            ),
        )


def test_partial_panel_defaults_missing_panel_type_without_mutating_caller(panel_df):
    metadata = AntibodyPanelMetadata(name="legacy-panel", version="0.0.0")
    assert metadata.panel_type is None

    panel = PartialPNAAntibodyPanel(panel_df, metadata)

    assert panel.metadata.panel_type == PanelType.PARTIAL
    assert metadata.panel_type is None
    assert panel.metadata is not metadata


def test_antibody_panel_metadata_from_adata_rejects_incomplete_schema():
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=["m1"]),
    )
    adata.uns["num_partial_panels"] = 2
    adata.uns["panel_metadata__0"] = {"name": "a", "version": "0.0.0"}

    with pytest.raises(KeyError, match="missing the metadata for panel at index 1"):
        AntibodyPanelMetadata.from_adata(adata)


def test_combination_from_adata_rejects_missing_panel_df(panel, hashing_panel):
    from pixelator.pna.anndata import add_panel_information

    combo = PNAAntibodyPanelCombination([panel.partial_panels()[0], hashing_panel])
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=["MarkerA"]),
    )
    adata = add_panel_information(adata, combo)
    del adata.uns["panel_df__1"]

    with pytest.raises(KeyError, match="missing the panel dataframe"):
        PNAAntibodyPanelCombination.from_adata(adata)


def test_antibody_panel_metadata_from_adata_reads_partial_panels():
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=["m1"]),
    )
    adata.uns["num_partial_panels"] = 1
    adata.uns["panel_metadata__0"] = {"name": "a", "version": "0.0.0"}

    metadatas = AntibodyPanelMetadata.from_adata(adata)
    assert len(metadatas) == 1
    assert metadatas[0].name == "a"


def test_legacy_panel_metadata_roundtrip(panel_df):
    metadata = AntibodyPanelMetadata(name="legacy-panel", version="1.0.0")
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=panel_df.copy(),
    )
    adata.uns["panel_metadata"] = {
        **metadata.model_dump(),
        "panel_columns": list(panel_df.columns),
    }

    combo = PNAAntibodyPanelCombination.from_adata(adata)
    assert combo.num_partial_panels == 1
    assert combo.name == "legacy-panel"


def test_upgrade_adata_migrates_legacy_panel_metadata(panel_df):
    """Patch bump migrates legacy uns keys to the multi-panel layout."""
    meta_old = AntibodyPanelMetadata(
        name="legacy-panel", version="1.0.0", product="test-product"
    )
    meta_new = AntibodyPanelMetadata(
        name="legacy-panel", version="1.0.1", product="test-product"
    )
    panel_old = PartialPNAAntibodyPanel(panel_df.copy(), meta_old)
    df_new = panel_df.copy()
    df_new.loc["marker1", "uniprot_id"] = "Q9UPN0"
    panel_new = PartialPNAAntibodyPanel(df_new, meta_new)

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=panel_df.copy(),
    )
    adata.uns["panel_metadata"] = {
        **meta_old.to_dict(),
        "panel_columns": list(panel_df.columns),
    }

    upgraded = PNAAntibodyPanelDiff(panel_old, panel_new).upgrade_adata(adata)

    assert "panel_metadata" not in upgraded.uns
    assert "panel_columns" not in upgraded.uns
    assert upgraded.uns["num_partial_panels"] == 1
    assert upgraded.uns["panel_metadata__0"]["version"] == "1.0.1"
    assert upgraded.var.loc["marker1", "uniprot_id"] == "Q9UPN0"
    reconstructed = PNAAntibodyPanelCombination.from_adata(upgraded)
    assert reconstructed.version == "1.0.1"
    assert reconstructed.num_partial_panels == 1


def test_upgrade_adata_maps_annotations_by_clone_identity():
    """Patch upgrades must keep each clone's panel_2 annotations on that clone.

    ``upgrade_adata`` writes join output into ``adata.var`` by ``panel_1`` row
    position. A Polars full join does not preserve that order unless
    ``maintain_order='left'``. This test gives ``panel_2`` the same clones in
    reverse row order, each with a distinct updated ``note`` and added
    ``extra`` value, and checks that marker *i* still receives ``new{i}`` /
    ``extra{i}`` instead of another clone's annotations.
    """
    sequences = [
        "ATCGAA",
        "GCTAAA",
        "ATCCAA",
        "TTTTAA",
        "AAAAAA",
        "CCCCAA",
        "GGGGAA",
        "TATAAA",
    ]
    uniprot_ids = [
        "P61769",
        "P05107",
        "P15391",
        "Q9UPN0",
        "P12345",
        "P56890",
        "P65470",
        "Q8WWI5",
    ]
    panel_df = pd.DataFrame(
        {
            "marker_id": [f"marker{i}" for i in range(1, 9)],
            "uniprot_id": uniprot_ids,
            "control": [False] * 8,
            "note": [f"old{i}" for i in range(1, 9)],
            "sequence_1": sequences,
            "sequence_2": sequences,
        }
    ).set_index("marker_id")
    df_new = panel_df.copy().iloc[::-1]
    df_new["note"] = [f"new{marker[-1]}" for marker in df_new.index]
    df_new["extra"] = [f"extra{marker[-1]}" for marker in df_new.index]

    meta_old = AntibodyPanelMetadata(
        name="clone-order-panel", version="1.0.0", product="test-product"
    )
    meta_new = AntibodyPanelMetadata(
        name="clone-order-panel", version="1.0.1", product="test-product"
    )
    panel_old = PartialPNAAntibodyPanel(panel_df.copy(), meta_old)
    panel_new = PartialPNAAntibodyPanel(df_new, meta_new)

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=panel_df.copy(),
    )
    adata.uns["panel_metadata"] = {
        **meta_old.to_dict(),
        "panel_columns": list(panel_df.columns),
    }

    upgraded = PNAAntibodyPanelDiff(panel_old, panel_new).upgrade_adata(adata)

    for i in range(1, 9):
        assert upgraded.var.loc[f"marker{i}", "note"] == f"new{i}"
        assert upgraded.var.loc[f"marker{i}", "extra"] == f"extra{i}"


def test_upgrade_adata_skips_hashing_rows_when_collapsed(panel, hashing_panel):
    """Collapsed sample-calling files keep hashing panels in uns, not in var."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    hashing_new_df = hashing_panel.df.copy()
    hashing_new_df["note"] = "updated"
    hashing_new = PNASampleHashingPanel(
        hashing_new_df,
        hashing_panel.metadata.model_copy(update={"version": "0.1.1"}),
    )

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(list(combo.markers), name="marker_id")),
    )
    adata = add_panel_information(adata, combo)
    adata = adata[:, list(base.markers)].copy()
    set_sample_calling_collapsed(adata, True)

    uncollapsed = add_panel_information(
        AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=pd.Index(list(base.markers), name="marker_id")),
        ),
        combo,
    )
    set_sample_calling_collapsed(uncollapsed, False)
    with pytest.raises(ValueError, match="missing panel clones"):
        PNAAntibodyPanelDiff(hashing_panel, hashing_new).upgrade_adata(uncollapsed)

    upgraded = PNAAntibodyPanelDiff(hashing_panel, hashing_new).upgrade_adata(adata)
    assert "HM-1" not in upgraded.var.index
    reconstructed = PNAAntibodyPanelCombination.from_adata(upgraded)
    assert reconstructed.hashing_panels is not None
    assert reconstructed.hashing_panels[0].version == "0.1.1"
    assert reconstructed.hashing_panels[0].df.loc["HM-1", "note"] == "updated"


def test_upgrade_adata_skips_hashing_rows_when_collapsed_inferred(panel, hashing_panel):
    """Legacy sample-called files without the collapsed flag still skip hashing clones."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    hashing_new_df = hashing_panel.df.copy()
    hashing_new_df["note"] = "updated"
    hashing_new = PNASampleHashingPanel(
        hashing_new_df,
        hashing_panel.metadata.model_copy(update={"version": "0.1.1"}),
    )

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(list(combo.markers), name="marker_id")),
    )
    adata = add_panel_information(adata, combo)
    adata = adata[:, list(base.markers)].copy()
    assert "sample_calling" not in adata.uns
    assert sample_calling_hashing_collapsed(adata) is True

    upgraded = PNAAntibodyPanelDiff(hashing_panel, hashing_new).upgrade_adata(adata)
    assert "HM-1" not in upgraded.var.index
    reconstructed = PNAAntibodyPanelCombination.from_adata(upgraded)
    assert reconstructed.hashing_panels is not None
    assert reconstructed.hashing_panels[0].version == "0.1.1"
    assert reconstructed.hashing_panels[0].df.loc["HM-1", "note"] == "updated"


def test_upgrade_adata_renames_collapsed_hashing_when_inferred_from_hash_counts(
    panel, hashing_panel
):
    """original_hash_counts_* infers collapsed layout for hashing id remaps."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    hashing_new = _renamed_hashing_panel(
        hashing_panel, {"HM-1": "NEW-1", "HM-2": "NEW-2"}
    )

    adata = AnnData(
        obs=pd.DataFrame(
            {
                "original_hash_counts_HM-1": [1.0],
                "original_hash_counts_HM-2": [2.0],
            },
            index=["c1"],
        ),
        var=pd.DataFrame(index=pd.Index(list(base.markers) + ["HM"], name="marker_id")),
    )
    adata = add_panel_information(adata, combo)
    assert sample_calling_hashing_collapsed(adata) is True

    upgraded = PNAAntibodyPanelDiff(hashing_panel, hashing_new).upgrade_adata(adata)
    assert "HM" not in upgraded.var.index
    assert "NEW" in upgraded.var.index
    assert "original_hash_counts_NEW-1" in upgraded.obs.columns
    assert upgraded.obs["original_hash_counts_NEW-1"].tolist() == [1.0]


def test_upgrade_adata_raises_when_collapsed_but_non_hashing_clone_missing(
    panel, hashing_panel
):
    """Collapsed upgrades may skip hashing clones, not missing non-hashing markers."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    base_new = type(base)(
        base.df.assign(note="updated"),
        base.metadata.model_copy(update={"version": "0.1.1"}),
    )
    remaining = [marker for marker in base.markers if marker != "MarkerA"]
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(remaining, name="marker_id")),
    )
    adata = add_panel_information(adata, combo)
    set_sample_calling_collapsed(adata, True)

    with pytest.raises(ValueError, match="missing panel clones"):
        PNAAntibodyPanelDiff(base, base_new).upgrade_adata(adata)

    inferred = add_panel_information(
        AnnData(
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(index=pd.Index(remaining, name="marker_id")),
        ),
        combo,
    )
    with pytest.raises(ValueError, match="missing panel clones"):
        PNAAntibodyPanelDiff(base, base_new).upgrade_adata(inferred)


def _renamed_hashing_panel(
    hashing_panel: PNASampleHashingPanel, mapping: dict[str, str]
):
    return PNASampleHashingPanel(
        hashing_panel.df.copy().rename(index=mapping),
        hashing_panel.metadata.model_copy(update={"version": "0.1.1"}),
    )


def test_upgrade_adata_renames_collapsed_hashing_marker_and_hash_counts(
    panel, hashing_panel
):
    """Hashing id bumps update collapsed var names and original_hash_counts_*."""
    base = panel.partial_panels()[0]
    combo = PNAAntibodyPanelCombination([base, hashing_panel])
    hashing_new = _renamed_hashing_panel(
        hashing_panel, {"HM-1": "NEW-1", "HM-2": "NEW-2"}
    )

    adata = AnnData(
        obs=pd.DataFrame(
            {
                "original_hash_counts_HM-1": [1.0],
                "original_hash_counts_HM-2": [2.0],
            },
            index=["c1"],
        ),
        var=pd.DataFrame(index=pd.Index(list(base.markers) + ["HM"], name="marker_id")),
    )
    adata = add_panel_information(adata, combo)
    set_sample_calling_collapsed(adata, True)

    diff = PNAAntibodyPanelDiff(hashing_panel, hashing_new)
    assert diff.collapsed_hashing_marker_id_mapping() == {"HM": "NEW"}
    assert diff.edgelist_marker_id_mapping(collapsed=True)["HM"] == "NEW"
    assert diff.edgelist_marker_id_mapping(collapsed=True)["HM-1"] == "NEW-1"

    upgraded = diff.upgrade_adata(adata)
    assert "HM" not in upgraded.var.index
    assert "NEW" in upgraded.var.index
    assert "original_hash_counts_HM-1" not in upgraded.obs.columns
    assert "original_hash_counts_NEW-1" in upgraded.obs.columns
    assert upgraded.obs["original_hash_counts_NEW-1"].tolist() == [1.0]
    assert upgraded.obs["original_hash_counts_NEW-2"].tolist() == [2.0]
    reconstructed = PNAAntibodyPanelCombination.from_adata(upgraded)
    assert reconstructed.hashing_panels is not None
    assert list(reconstructed.hashing_panels[0].df.index) == ["NEW-1", "NEW-2"]

    edgelist = pl.DataFrame({"marker_1": ["HM", "MarkerA"], "marker_2": ["HM", "HM-1"]})
    collapsed_edges = _apply_edgelist_marker_id_mapping(diff, edgelist, collapsed=True)
    assert collapsed_edges["marker_1"].to_list() == ["NEW", "MarkerA"]
    assert collapsed_edges["marker_2"].to_list() == ["NEW", "NEW-1"]


def test_upgrade_adata_rejects_hashing_hash_group_change(panel, hashing_panel):
    """Hashing patch bumps cannot change the -<digits> suffix."""
    hashing_new = _renamed_hashing_panel(hashing_panel, {"HM-1": "HM-9"})
    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(list(panel.markers), name="marker_id")),
    )
    adata = add_panel_information(
        adata, PNAAntibodyPanelCombination([panel.partial_panels()[0], hashing_panel])
    )
    set_sample_calling_collapsed(adata, True)
    with pytest.raises(ValueError, match="hash group suffix"):
        PNAAntibodyPanelDiff(hashing_panel, hashing_new).upgrade_adata(adata)


def test_upgrade_adata_rejects_inconsistent_hashing_base_rename(hashing_panel):
    """All hashing ids that collapse to the same base must keep one new base."""
    hashing_new = _renamed_hashing_panel(hashing_panel, {"HM-1": "NEW-1"})
    with pytest.raises(ValueError, match="multiple names"):
        PNAAntibodyPanelDiff(
            hashing_panel, hashing_new
        ).collapsed_hashing_marker_id_mapping()


def _v2_style_b2m_panel(
    *, version: str = "0.1.0", rename: dict[str, str] | None = None
):
    """Non-hashing B2M plus hashing B2M-1/B2M-2 on one CSV (v2 layout)."""
    df = pd.DataFrame(
        {
            "marker_id": ["B2M", "CD3", "B2M-1", "B2M-2"],
            "control": [False, False, False, False],
            "sequence_1": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sequence_2": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sample_hashing": [False, False, True, True],
        }
    ).set_index("marker_id")
    if rename:
        df = df.rename(index=rename)
    return PartialPNAAntibodyPanel(
        df,
        AntibodyPanelMetadata(name="v2-b2m", version=version, product="v2-b2m"),
    )


def test_hashing_base_rename_requires_matching_non_hashing_marker():
    """Hashing-only B2M rename would clobber the non-hashing B2M row after collapse."""
    panel_old = _v2_style_b2m_panel()
    panel_new = _v2_style_b2m_panel(
        version="0.1.1",
        rename={"B2M-1": "NEWB2M-1", "B2M-2": "NEWB2M-2"},
    )
    with pytest.raises(ValueError, match="must rename together"):
        PNAAntibodyPanelDiff(panel_old, panel_new).collapsed_hashing_marker_id_mapping()


def test_non_hashing_base_rename_requires_matching_hashing_markers():
    """Non-hashing-only B2M rename must move the hashing family as well."""
    panel_old = _v2_style_b2m_panel()
    panel_new = _v2_style_b2m_panel(version="0.1.1", rename={"B2M": "NEWB2M"})
    with pytest.raises(ValueError, match="must rename together"):
        PNAAntibodyPanelDiff(panel_old, panel_new).collapsed_hashing_marker_id_mapping()


def test_hashing_and_non_hashing_base_rename_together_on_collapsed_adata():
    """B2M hashing and non-hashing names renamed together apply to the remaining row after collapse."""
    panel_old = _v2_style_b2m_panel()
    panel_new = _v2_style_b2m_panel(
        version="0.1.1",
        rename={"B2M": "NEWB2M", "B2M-1": "NEWB2M-1", "B2M-2": "NEWB2M-2"},
    )
    diff = PNAAntibodyPanelDiff(panel_old, panel_new)
    assert diff.collapsed_hashing_marker_id_mapping() == {"B2M": "NEWB2M"}
    assert diff.marker_id_mapping()["B2M"] == "NEWB2M"

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["B2M", "CD3"], name="marker_id")),
    )
    adata = add_panel_information(adata, PNAAntibodyPanelCombination(panel_old))
    set_sample_calling_collapsed(adata, True)
    upgraded = diff.upgrade_adata(adata)
    assert "B2M" not in upgraded.var.index
    assert "NEWB2M" in upgraded.var.index
    assert "CD3" in upgraded.var.index


def _b2m_base_panel(*, version: str = "0.1.0", rename: dict[str, str] | None = None):
    df = pd.DataFrame(
        {
            "marker_id": ["B2M", "CD3"],
            "control": [False, False],
            "sequence_1": ["AAAAAA", "CCCCCC"],
            "sequence_2": ["AAAAAA", "CCCCCC"],
        }
    ).set_index("marker_id")
    if rename:
        df = df.rename(index=rename)
    return PNABasePanel(
        df,
        AntibodyPanelMetadata(
            name="b2m-base",
            version=version,
            product="b2m-base",
            panel_type=PanelType.BASE,
        ),
    )


def _b2m_hashing_member(
    *, version: str = "0.1.0", rename: dict[str, str] | None = None
):
    df = pd.DataFrame(
        {
            "marker_id": ["B2M-1", "B2M-2"],
            "control": [False, False],
            "sequence_1": ["GGGGGG", "TTTTTT"],
            "sequence_2": ["GGGGGG", "TTTTTT"],
            "sample_hashing": [True, True],
        }
    ).set_index("marker_id")
    if rename:
        df = df.rename(index=rename)
    return PNASampleHashingPanel(
        df,
        AntibodyPanelMetadata(
            name="b2m-hash",
            version=version,
            product="b2m-hash",
            panel_type=PanelType.SAMPLE_HASHING,
        ),
    )


def test_upgrade_adata_rejects_hashing_member_rename_that_clobbers_non_hashing_b2m():
    """A hashing-member bump still sees non-hashing B2M on the stored combination."""
    base = _b2m_base_panel()
    hashing = _b2m_hashing_member()
    hashing_new = _b2m_hashing_member(
        version="0.1.1", rename={"B2M-1": "NEWB2M-1", "B2M-2": "NEWB2M-2"}
    )
    assert PNAAntibodyPanelDiff(
        hashing, hashing_new
    ).collapsed_hashing_marker_id_mapping() == {"B2M": "NEWB2M"}

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["B2M", "CD3"], name="marker_id")),
    )
    adata = add_panel_information(adata, PNAAntibodyPanelCombination([base, hashing]))
    set_sample_calling_collapsed(adata, True)
    with pytest.raises(ValueError, match="must rename together"):
        PNAAntibodyPanelDiff(hashing, hashing_new).upgrade_adata(adata)


def test_upgrade_adata_rejects_non_hashing_rename_that_leaves_hashing_b2m_behind():
    """A base-panel B2M rename must not leave hashing clones collapsing to B2M."""
    base = _b2m_base_panel()
    hashing = _b2m_hashing_member()
    base_new = _b2m_base_panel(version="0.1.1", rename={"B2M": "NEWB2M"})

    adata = AnnData(
        obs=pd.DataFrame(index=["c1"]),
        var=pd.DataFrame(index=pd.Index(["B2M", "CD3"], name="marker_id")),
    )
    adata = add_panel_information(adata, PNAAntibodyPanelCombination([base, hashing]))
    set_sample_calling_collapsed(adata, True)
    with pytest.raises(ValueError, match="must rename together"):
        PNAAntibodyPanelDiff(base, base_new).upgrade_adata(adata)


def _hm_and_b2m_panel(*, version: str = "0.1.0", rename: dict[str, str] | None = None):
    """Non-hashing B2M plus hashing HM-1/HM-2 (distinct collapsed names)."""
    df = pd.DataFrame(
        {
            "marker_id": ["B2M", "HM-1", "HM-2"],
            "control": [False, False, False],
            "sequence_1": ["AAAAAA", "CCCCCC", "GGGGGG"],
            "sequence_2": ["AAAAAA", "CCCCCC", "GGGGGG"],
            "sample_hashing": [False, True, True],
        }
    ).set_index("marker_id")
    if rename:
        df = df.rename(index=rename)
    return PartialPNAAntibodyPanel(
        df,
        AntibodyPanelMetadata(name="hm-b2m", version=version, product="hm-b2m"),
    )


def test_hashing_rename_rejects_collapsed_name_collision_with_non_hashing():
    """HM-1 → B2M-1 must not rewrite non-hashing B2M after collapse."""
    panel_old = _hm_and_b2m_panel()
    panel_new = _hm_and_b2m_panel(
        version="0.1.1", rename={"HM-1": "B2M-1", "HM-2": "B2M-2"}
    )
    with pytest.raises(ValueError, match="collides with"):
        PNAAntibodyPanelDiff(panel_old, panel_new).collapsed_hashing_marker_id_mapping()


def test_hashing_families_cannot_collapse_to_the_same_name():
    """Two hashing families must not share a collapsed name after a bump."""
    df = pd.DataFrame(
        {
            "marker_id": ["HM-1", "HM-2", "XX-3", "XX-4"],
            "control": [False, False, False, False],
            "sequence_1": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sequence_2": ["AAAAAA", "CCCCCC", "GGGGGG", "TTTTTT"],
            "sample_hashing": [True, True, True, True],
        }
    ).set_index("marker_id")
    hashing_old = PNASampleHashingPanel(
        df,
        AntibodyPanelMetadata(
            name="two-fam",
            version="0.1.0",
            product="two-fam",
            panel_type=PanelType.SAMPLE_HASHING,
        ),
    )
    hashing_new = PNASampleHashingPanel(
        df.rename(
            index={
                "HM-1": "NEW-1",
                "HM-2": "NEW-2",
                "XX-3": "NEW-3",
                "XX-4": "NEW-4",
            }
        ),
        hashing_old.metadata.model_copy(update={"version": "0.1.1"}),
    )
    with pytest.raises(ValueError, match="collapse to the same name"):
        PNAAntibodyPanelDiff(
            hashing_old, hashing_new
        ).collapsed_hashing_marker_id_mapping()


def test_edgelist_marker_id_mapping_renames_marker_columns(panel_df):
    """Patch bumps that rename marker_id rewrite edgelist marker columns."""
    meta_old = AntibodyPanelMetadata(
        name="rename-panel", version="1.0.0", product="test-product"
    )
    meta_new = AntibodyPanelMetadata(
        name="rename-panel", version="1.0.1", product="test-product"
    )
    panel_old = PartialPNAAntibodyPanel(panel_df.copy(), meta_old)
    df_new = panel_df.copy().rename(index={"marker1": "marker1-renamed"})
    panel_new = PartialPNAAntibodyPanel(df_new, meta_new)

    edgelist = pl.DataFrame(
        {
            "marker_1": ["marker1", "marker2"],
            "marker_2": ["marker3", "marker1"],
            "component": ["c1", "c1"],
        }
    )
    diff = PNAAntibodyPanelDiff(panel_old, panel_new)
    assert diff.marker_id_mapping() == {"marker1": "marker1-renamed"}
    upgraded = _apply_edgelist_marker_id_mapping(diff, edgelist)
    assert upgraded["marker_1"].to_list() == ["marker1-renamed", "marker2"]
    assert upgraded["marker_2"].to_list() == ["marker3", "marker1-renamed"]


def test_panel_from_pxl(pxl_file):
    """Verify panel from pxl.

    Args:
        pxl_file: pxl file.
    """
    panel = PNAAntibodyPanelCombination.from_pxl_dataset(read(pxl_file))
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.1.0"
    assert panel.description == "Test R&D panel for PNA"
    assert panel.aliases == ["test-pna"]
    assert panel.filename == Path(pxl_file).name
    assert panel.filepath == Path(pxl_file).resolve()

    expected_data = {
        "marker_id": ["MarkerA", "MarkerB", "MarkerC"],
        "control": [False, False, True],
        "uniprot_id": ["P12345", "P56890;P65470", ""],
        "sequence_1": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
        "sequence_2": ["ACTTCCTAGG", "CCAGGTTCCG", "CAGCTATGGT"],
    }
    expected_df = pd.DataFrame(expected_data).set_index("marker_id")
    assert_frame_equal(
        panel.df.drop(columns=["partial_panel_name", "partial_panel_type"]),
        expected_df,
    )


def test_panel_header_trailing_commas_warns_and_recovers(caplog):
    """Verify panel header trailing commas warns and recovers.

    Args:
        caplog: caplog.
    """
    panel_content = """# ---
# name: test-pna-panel,
# product: test-product,
# aliases:
#   - test-pna
# description: Test R&D panel for PNA,
# version: 1.0.0,
# ---
marker_id,control,sequence_1,sequence_2
MarkerA,no,ACTTCCTAGG,ACTTCCTAGG
"""
    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(panel_content)
        tmp_file.flush()

        with caplog.at_level("WARNING"):
            panel = PartialPNAAntibodyPanel.from_csv(tmp_file.name)

    assert panel.name == "test-pna-panel"
    assert panel.version == "1.0.0"
    assert panel.filepath == Path(tmp_file.name).resolve()
    assert "trailing comma" in caplog.text.lower()


def test_panel_header_non_recoverable_yaml_still_fails():
    """Verify panel header non recoverable yaml still fails."""
    panel_content = """# ---
# name: test panel
# aliases: [test-alias
# version: 0.1.0
# ---
marker_id,control,nuclear,sequence,conj_id
CD45,no,no,TCCCTTGCGATTTAC,test001
"""
    with NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(panel_content)
        tmp_file.flush()

        with pytest.raises(yaml.YAMLError):
            PartialPNAAntibodyPanel.from_csv(tmp_file.name)
