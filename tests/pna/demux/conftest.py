"""Copyright © 2025 Pixelgen Technologies AB."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def demux_m1_arrow_data(pna_data_root):
    return (
        pna_data_root
        / "intermediate-demux-results"
        / "PNA055_Sample07_filtered_S7.demux.m1.part_000.arrow"
    )


@pytest.fixture()
def demux_m2_arrow_data(pna_data_root):
    return (
        pna_data_root
        / "intermediate-demux-results"
        / "PNA055_Sample07_filtered_S7.demux.m2.part_000.arrow"
    )


@pytest.fixture()
def demux_intermediary_dir(tmpdir, demux_m1_arrow_data, demux_m2_arrow_data) -> Path:
    shutil.copy(demux_m1_arrow_data, tmpdir)
    shutil.copy(demux_m2_arrow_data, tmpdir)
    return Path(tmpdir)
