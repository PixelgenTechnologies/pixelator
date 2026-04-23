"""
Tests for config module

Copyright © 2023 Pixelgen Technologies AB.
"""

import copy

import pandas as pd
import pytest

from pixelator.common.config import (
    RegionType,
    get_position_in_parent,
)
from pixelator.pna.config.config_class import (
    PNAConfig,
    load_assays_package,
    load_panels_package,
)
from pixelator.pna.config.config_instance import pna_config
from pixelator.pna.config.panel import PNAAntibodyPanel, load_antibody_panel


def test_config_creation():
    config = PNAConfig()
    load_assays_package(config, "pixelator.pna.resources.assays")

    assert {"proxiome-v1"}.issubset(config.assays)

    assay = config.get_assay("proxiome-v1")
    assert assay.name == "proxiome-v1"


def test_load_assays_dir(pna_data_root):
    config = PNAConfig()
    config.load_assays(pna_data_root / "assays")

    a1 = config.get_assay("test-pna-assay")
    assert a1.name == "test-pna-assay"


def test_assay_region_ids():
    all_region_ids = pna_config.get_assay("proxiome-v1").region_ids

    expected_region_ids = {
        "amplicon",
        "pid-1",
        "pid-2",
        "umi-1",
        "umi-2",
        "lbs-1",
        "lbs-2",
    }

    assert expected_region_ids.issubset(all_region_ids)


def test_assay_get_region_by_id():
    assay = pna_config.get_assay("proxiome-v1")

    region = assay.get_region_by_id("pid-2")
    assert region.region_id == "pid-2"


def test_assay_get_regions_by_type():
    assay = pna_config.get_assay("proxiome-v1")

    regions = assay.get_regions_by_type("lbs-1")
    for r in regions:
        assert r.region_type is RegionType.LBS


def test_get_position_in_amplicon_pna_1():
    design = pna_config.get_assay("proxiome-v1")

    umi1_pos = get_position_in_parent(design, "umi-1")
    pid1_pos = get_position_in_parent(design, "pid-1")
    lbs1_pos = get_position_in_parent(design, "lbs-1")
    uei_pos = get_position_in_parent(design, "uei")
    lbs2_pos = get_position_in_parent(design, "lbs-2")
    umi2_pos = get_position_in_parent(design, "umi-2")
    pid2_pos = get_position_in_parent(design, "pid-2")

    assert umi1_pos == (0, 28)
    assert pid1_pos == (28, 38)
    assert lbs1_pos == (38, 70)
    assert uei_pos == (70, 85)
    assert lbs2_pos == (85, 103)
    assert pid2_pos == (103, 113)
    assert umi2_pos == (113, 141)


@pytest.fixture()
def config_with_multiple_versions(pna_data_root):
    """Create a config fixture with multiple versions of the same panel.

    Args:
        pna_data_root: Root path to PNA test data files.

    Returns:
        A PNA config object populated with panel version variants.

    """
    new_config = copy.deepcopy(pna_config)
    new_config = load_panels_package(new_config, "tests.pna.data.panels")
    new_config.load_panel_file(pna_data_root / "test-pna-panel-v1.1.0.csv")
    new_config.load_panel_file(pna_data_root / "test-pna-panel-v2.0.0.csv")
    return new_config


def test_loading_panel_from_config(config_with_multiple_versions):
    """Verify resolving a panel by inline exact version specifier.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    panel_name = "test-pna-panel==1.1.0"
    panel = config_with_multiple_versions.get_panel(panel_name)
    assert panel.name == "test-pna-panel"
    assert panel.version == "1.1.0"


def test_loading_multiple_minor_version(config_with_multiple_versions):
    """Verify ambiguous minor-version requests raise a descriptive error.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    panel_name = "test-pna-panel==1"
    with pytest.raises(
        ValueError,
        match=f"Multiple minor versions found for panel {panel_name}. "
        + "Refusing to automatically select the latest out of multiple minor versions. "
        + "Minor versions usually means there was a change in clones used for one or "
        + "more markers. Panels might not be fully compatible!\n"
        + "Please specify the minor version in the panel name or "
        + "alias to disambiguate.",
    ):
        config_with_multiple_versions.get_panel(panel_name)


def test_loading_multiple_major_version(config_with_multiple_versions):
    """Verify ambiguous major-version requests raise a descriptive error.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    panel_name = "test-pna-panel>=0.0.1"
    with pytest.raises(
        ValueError,
        match=f"Multiple major versions found for panel {panel_name}. Please specify the major and minor version in the panel name or alias to disambiguate.",
    ):
        config_with_multiple_versions.get_panel(panel_name)


@pytest.mark.parametrize(
    "panel_alias,panel_name,panel_version",
    [
        ("test-pna", "test-pna-panel", "0.2.0"),
    ],
)
def test_loading_panel_from_config_alias(
    config_with_multiple_versions, panel_alias, panel_name, panel_version
):
    panel = config_with_multiple_versions.get_panel(panel_alias)
    assert panel.name == panel_name
    assert panel.version == panel_version


def test_loading_panel_from_config_specific_version(config_with_multiple_versions):
    """Verify resolving a named panel using the explicit version argument.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    panel = config_with_multiple_versions.get_panel("test-pna-panel", version="1.1.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "1.1.0"

    panel = config_with_multiple_versions.get_panel("test-pna-panel", version="2.0.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "2.0.0"


def test_loading_panel_from_config_product_and_specific_version(
    config_with_multiple_versions,
):
    """Verify resolving a panel by product name plus explicit version.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    panel = config_with_multiple_versions.get_panel("test-product", version="1.1.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "1.1.0"


def test_loading_panel_from_config_alias_and_specific_version(
    config_with_multiple_versions,
):
    """Verify alias lookup can still be filtered by explicit version.

    Args:
        config_with_multiple_versions: Config fixture with multiple panel versions.

    """
    # NOTE: Panel aliases is a DEPRECATED feature and should not be used for new panels.
    #
    # When trying to specify the version using "v1", "v3" etc within the panel_name, alias or
    # product name, as shown below, this is still only interpretated as part of the name.
    #
    # For example the alias "test-pna-panel-v3" points to the panel_name "test-pna-panel" which has
    # versions 1.0.0, 1.1.0 and 2.0.0 available.
    #
    # When used with an alias this leads to a somewhat weird situation where, in this case, the
    # alias "test-pna-panel-v3" can be used to load any of the versions of the panel even though the
    # alias was specified in the 2.0.0 version of the panel.
    # This was accepted behaviour as the alias was mainly be used for backwards compability
    # when a panel had been renamed and as such points to panel_name and not a specific version.
    #
    # For new panels use the panel_name alternatively the product name together with the version
    # speciefied as a parameter to config.get_panel or appended to the end of the product name.

    panel = config_with_multiple_versions.get_panel("test-pna-v3", version="1.1.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "1.1.0"


def test_load_antibody_panel_util(pna_data_root):
    """Verify utility loading works for config names and filesystem paths.

    Args:
        pna_data_root: Root path to PNA test data files.

    """
    cgf_panel = load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")
    assert cgf_panel.name == "proxiome-v1-immuno-155-v1.0"

    path_panel = load_antibody_panel(
        pna_config, pna_data_root / "test-pna-panel-v1.1.0.csv"
    )
    assert path_panel.name == "test-pna-panel"
    assert path_panel.filename == "test-pna-panel-v1.1.0.csv"

    with pytest.raises(AssertionError):
        load_antibody_panel(pna_config, "human-qwdqwdqwdqdw-proteomics")


def test_panel_with_non_dna_sequences(pna_data_root):
    """Verify non-DNA sequence content yields validation errors.

    Args:
        pna_data_root: Root path to PNA test data files.

    """
    panel_df = pd.read_csv(
        pna_data_root / "test-pna-panel-v1.1.0.csv", index_col="marker_id", comment="#"
    ).fillna("")
    panel_df["control"] = panel_df["control"].map(lambda s: s.lower() == "yes")
    panel_df.loc["CD45", "sequence_1"] = "PPPPPP"
    errors = PNAAntibodyPanel.validate_antibody_panel(panel_df)
    assert len(errors) == 2
    assert errors[0] == "All sequence_1 values must have the same length."
    assert (
        errors[1]
        == "All sequence_1 values must only contain ATCG characters. Offending values: ['PPPPPP']"
    )


def test_list_panel_names(pna_data_root):
    assert sorted(pna_config.list_panel_names(include_aliases=True)) == sorted(
        [
            "proxiome-v1-immuno-155-v1.0",
            "proxiome-v1-immuno-155-v1.1",
            "proxiome-v1-immuno-156-FLAG-v1.0",
            "proxiome-v1-immuno-156-FLAG-v1.1",
            "proxiome-v1-immuno-156-FMC63-v1.0",
            "proxiome-v1-immuno-156-FMC63-v1.1",
            "proxiome-v2-immuno-155-v1.0",
        ]
    )

    assert sorted(pna_config.list_panel_names(include_aliases=False)) == [
        "proxiome-v1-immuno-155-v1.0",
        "proxiome-v1-immuno-155-v1.1",
        "proxiome-v1-immuno-156-FLAG-v1.0",
        "proxiome-v1-immuno-156-FLAG-v1.1",
        "proxiome-v1-immuno-156-FMC63-v1.0",
        "proxiome-v1-immuno-156-FMC63-v1.1",
        "proxiome-v2-immuno-155-v1.0",
    ]


def test_loading_duplicate_aliases(config_with_multiple_versions, pna_data_root):
    this_config = copy.deepcopy(config_with_multiple_versions)
    from pixelator.common.config.config_class import PanelException

    with pytest.raises(PanelException):
        this_config.load_panel_file(
            pna_data_root / "test-pna-panel-duplicate-aliases.csv"
        )
