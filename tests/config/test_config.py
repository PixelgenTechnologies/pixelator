"""
Tests for config module

Copyright (c) 2023 Pixelgen Technologies AB.
"""
import copy

import pytest

from pixelator.config import (
    Config,
    RegionType,
    config,
    get_position_in_parent,
    load_assays_package,
)
from pixelator.config.panel import load_antibody_panel


def test_config_creation():
    config = Config()
    load_assays_package(config, "pixelator.resources.assays")

    assert {"D21"}.issubset(config.assays)

    assay = config.get_assay("D21")
    assert assay.name == "D21"


def test_parsing_recursion_protection(data_root):
    cfg = Config()

    with pytest.raises(RecursionError):
        cfg.load_assay(data_root / "recursion_attack.yaml")


def test_assay_region_ids():
    all_region_ids = config.get_assay("D21").region_ids

    expected_region_ids = {
        "amplicon",
        "upi-b",
        "pbs-2",
        "upi-a",
        "pbs-1",
        "umi-b",
        "bc",
    }

    assert expected_region_ids.issubset(all_region_ids)


def test_assay_get_region_by_id():
    assay = config.get_assay("D21")

    region = assay.get_region_by_id("pbs-2")
    assert region.region_id == "pbs-2"


def test_assay_get_regions_by_type():
    assay = config.get_assay("D21")

    regions = assay.get_regions_by_type("pbs")
    for r in regions:
        assert r.region_type is RegionType.PBS


def test_get_position_in_amplicon_D21():
    design = config.get_assay("D21")

    pbs1_pos = get_position_in_parent(design, "pbs-1")
    pbs2_pos = get_position_in_parent(design, "pbs-2")
    umib_pos = get_position_in_parent(design, "umi-b")
    upia_pos = get_position_in_parent(design, "upi-a")
    upib_pos = get_position_in_parent(design, "upi-b")
    bc_pos = get_position_in_parent(design, "bc")

    assert pbs1_pos == (92, 114)
    assert pbs2_pos == (25, 67)
    assert umib_pos == (114, 124)
    assert upia_pos == (67, 92)
    assert upib_pos == (0, 25)
    assert bc_pos == (124, 132)


@pytest.fixture()
def config_with_multiple_versions(data_root):
    new_config = copy.copy(config)
    new_config.load_panel_file(data_root / "UNO_D21_Beta_old.csv")
    return new_config


def test_loading_panel_from_config(config_with_multiple_versions):
    panel = config_with_multiple_versions.get_panel(
        "human-sc-immunology-spatial-proteomics"
    )
    assert panel.name == "human-sc-immunology-spatial-proteomics"
    assert panel.version == "0.3.0"


def test_loading_panel_from_config_specific_version(config_with_multiple_versions):
    panel = config_with_multiple_versions.get_panel(
        "human-sc-immunology-spatial-proteomics", version="0.2.0"
    )
    assert panel.name == "human-sc-immunology-spatial-proteomics"
    assert panel.version == "0.2.0"


def test_load_antibody_panel_util(data_root):
    cgf_panel = load_antibody_panel(config, "human-sc-immunology-spatial-proteomics")
    assert cgf_panel.name == "human-sc-immunology-spatial-proteomics"

    path_panel = load_antibody_panel(config, data_root / "UNO_D21_Beta.csv")
    assert path_panel.name == "human-sc-immunology-spatial-proteomics"
    assert path_panel.filename == "UNO_D21_Beta.csv"

    with pytest.raises(AssertionError):
        load_antibody_panel(config, "human-qwdqwdqwdqdw-proteomics")
