"""Copyright © 2025 Pixelgen Technologies AB."""

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
    PNAAntibodyPanelCombination,
    PNABasePanel,
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
        df=panel_df,
        metadata=AntibodyPanelMetadata(**metadata),
        file_name="test.csv",
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
    assert panel.size == 3


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
    combo_hashing_first = PNAAntibodyPanelCombination.from_list_of_subpanels(
        [hashing_panel, base]
    )
    combo_base_first = PNAAntibodyPanelCombination.from_list_of_subpanels(
        [base, hashing_panel]
    )

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
        PNAAntibodyPanelCombination.from_list_of_subpanels(
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
        PNAAntibodyPanelCombination.from_list_of_subpanels(
            [
                PNABasePanel(panel_df, meta1),
                PNABasePanel(conflicting_df, meta2),
            ]
        )


def test_combination_aliases_raises_for_multi_panel(panel_df, hashing_panel):
    base = PNABasePanel(
        panel_df,
        AntibodyPanelMetadata(
            name="base-panel", version="0.0.0", panel_type=PanelType.BASE
        ),
    )
    combo = PNAAntibodyPanelCombination.from_list_of_subpanels([base, hashing_panel])

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
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
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
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
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
            df=panel_df,
            metadata=AntibodyPanelMetadata(
                name="mock-name",
                version="0.0.0",
            ),
        )


def test_panel_validation_ok_on_concatenated_uniprot_ids(panel_df):
    """Verify panel validation ok on concatenated uniprot ids.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = "P05107;P15391"
    PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="mock-name",
            version="0.0.0",
        ),
    )


def test_panel_validation_ok_uniprotid_empty(panel_df):
    """Verify panel validation ok uniprotid empty.

    Args:
        panel_df: panel df.
    """
    panel_df.loc["marker1", "uniprot_id"] = ""
    PNAAntibodyPanelCombination(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="mock-name",
            version="0.0.0",
        ),
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

    combo = PNAAntibodyPanelCombination.from_list_of_subpanels(
        [panel.partial_panels()[0], hashing_panel]
    )
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


def test_panel_from_pxl(pxl_file):
    """Verify panel from pxl.

    Args:
        pxl_file: pxl file.
    """
    panel = PNAAntibodyPanelCombination.from_pxl_dataset(read(pxl_file))
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.1.0"
    assert panel.description == "Test R&D panel for RNA"
    assert panel.aliases == ["test-pna"]

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
            panel = PNAAntibodyPanel.from_csv(tmp_file.name)

    assert panel.name == "test-pna-panel"
    assert panel.version == "1.0.0"
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
            PNAAntibodyPanel.from_csv(tmp_file.name)
