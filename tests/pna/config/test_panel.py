"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import pytest
import ruamel.yaml as yaml
from anndata import AnnData
from pandas.testing import assert_frame_equal

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import (
    PanelType,
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNAAntibodyPanelCombination,
    PNAAntibodyPanelDiff,
    PNABasePanel,
    PNASampleHashingPanel,
    sample_hashing_mask,
)
from pixelator.pna.pixeldataset import read


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
    hashing_col = combo.df["sample_hashing"]
    assert pd.api.types.is_bool_dtype(hashing_col)
    assert hashing_col.loc[["HM-1", "HM-2"]].all()
    assert not hashing_col.loc[panel_df.index].any()
    assert combo.df[sample_hashing_mask(hashing_col)].index.tolist() == ["HM-1", "HM-2"]


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
            "marker_id": ["extra"],
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


def test_base_panel_sets_panel_type_when_missing(panel_df):
    panel = PartialPNAAntibodyPanel(
        panel_df,
        AntibodyPanelMetadata(name="base-panel", version="0.0.0"),
    )
    assert panel.metadata.panel_type == PanelType.PARTIAL


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
