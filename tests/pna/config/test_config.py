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

    assert {"pna-2"}.issubset(config.assays)

    assay = config.get_assay("pna-2")
    assert assay.name == "pna-2"


def test_load_assays_dir(pna_data_root):
    config = PNAConfig()
    config.load_assays(pna_data_root / "assays")

    a1 = config.get_assay("test-pna-assay")
    assert a1.name == "test-pna-assay"


def test_assay_region_ids():
    all_region_ids = pna_config.get_assay("pna-2").region_ids

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
    assay = pna_config.get_assay("pna-2")

    region = assay.get_region_by_id("pid-2")
    assert region.region_id == "pid-2"


def test_assay_get_regions_by_type():
    assay = pna_config.get_assay("pna-2")

    regions = assay.get_regions_by_type("lbs-1")
    for r in regions:
        assert r.region_type is RegionType.LBS


def test_get_position_in_amplicon_pna_1():
    design = pna_config.get_assay("pna-2")

    umi1_pos = get_position_in_parent(design, "umi-1")
    pid1_pos = get_position_in_parent(design, "pid-1")
    lbs1_pos = get_position_in_parent(design, "lbs-1")
    uei_pos = get_position_in_parent(design, "uei")
    lbs2_pos = get_position_in_parent(design, "lbs-2")
    umi2_pos = get_position_in_parent(design, "umi-2")
    pid2_pos = get_position_in_parent(design, "pid-2")

    assert umi1_pos == (0, 28)
    assert pid1_pos == (28, 38)
    assert lbs1_pos == (38, 71)
    assert uei_pos == (71, 86)
    assert lbs2_pos == (86, 104)
    assert pid2_pos == (104, 114)
    assert umi2_pos == (114, 142)


@pytest.fixture()
def config_with_multiple_versions(pna_data_root):
    new_config = copy.deepcopy(pna_config)
    new_config = load_panels_package(new_config, "tests.pna.data.panels")
    new_config.load_panel_file(pna_data_root / "test-pna-panel-v2.csv")
    return new_config


def test_loading_panel_from_config(config_with_multiple_versions):
    panel = config_with_multiple_versions.get_panel("test-pna-panel")
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.2.0"


@pytest.mark.parametrize(
    "panel_name", ["proxiome-immuno-155-v1", "proxiome-immuno-155-v2"]
)
def test_loading_panel_from_config_alias(panel_name):
    panel = pna_config.get_panel(panel_name)
    assert panel.name == panel_name
    assert panel.version == f"{panel_name[-1]}.0.0"


def test_loading_panel_from_config_specific_version(config_with_multiple_versions):
    panel = config_with_multiple_versions.get_panel("test-pna-panel", version="0.1.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.1.0"

    panel = config_with_multiple_versions.get_panel("test-pna-panel", version="0.2.0")
    assert panel.name == "test-pna-panel"
    assert panel.version == "0.2.0"


def test_load_antibody_panel_util(pna_data_root):
    cgf_panel = load_antibody_panel(pna_config, "proxiome-immuno-155")
    assert cgf_panel.name == "proxiome-immuno-155-v2"

    path_panel = load_antibody_panel(
        pna_config, pna_data_root / "test-pna-panel-v2.csv"
    )
    assert path_panel.name == "test-pna-panel"
    assert path_panel.filename == "test-pna-panel-v2.csv"

    with pytest.raises(AssertionError):
        load_antibody_panel(pna_config, "human-qwdqwdqwdqdw-proteomics")


def test_panel_with_non_dna_sequences(pna_data_root):
    panel_df = pd.read_csv(pna_data_root / "test-pna-panel-v2.csv", skiprows=9)
    panel_df.loc[0, "sequence_1"] = "PPPPPP"
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
            "proxiome-immuno-155",
            "proxiome-immuno-155plex",
            "proxiome-immuno-155-v1",
            "proxiome-immuno-155plex-v1",
            "proxiome-immuno-155-v2",
            "proxiome-immuno-155plex-v2",
            "proxiome-immuno-156-FLAG",
            "proxiome-immuno-156-FLAG-v2",
            "proxiome-immuno-156-FLAGplex",
            "proxiome-immuno-156-FLAGplex-v2",
            "proxiome-immuno-156-FMC63",
            "proxiome-immuno-156-FMC63plex",
            "proxiome-immuno-156-FMC63-v1",
            "proxiome-immuno-156-FMC63-v2",
            "proxiome-immuno-156-FMC63plex-v1",
            "proxiome-immuno-156-FMC63plex-v2",
        ]
    )

    assert sorted(pna_config.list_panel_names(include_aliases=False)) == [
        "proxiome-immuno-155-v1",
        "proxiome-immuno-155-v2",
        "proxiome-immuno-156-FLAG-v2",
        "proxiome-immuno-156-FMC63-v1",
        "proxiome-immuno-156-FMC63-v2",
    ]


def test_loading_duplicate_aliases(pna_data_root):
    this_config = copy.deepcopy(pna_config)
    from pixelator.common.config.config_class import PanelException

    with pytest.raises(PanelException):
        this_config.load_panel_file(
            pna_data_root / "test-pna-panel-duplicate-aliases.csv"
        )
