"""Analysis tests for pna data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.pna.analysis.analysis import ProximityAnalysis
from pixelator.pna.analysis.proximity import calculate_differential_proximity
from pixelator.pna.analysis_engine import AnalysisManager
from pixelator.pna.pixeldataset import PNAPixelDataset
from pixelator.pna.pixeldataset.io import PxlFile

PROXIMITY_DATA = """component,marker_1,marker_2,join_count,join_count_expected_mean,join_count_expected_sd,join_count_z,join_count_pvalue
9864156ed5c9eb6c,MarkerA,MarkerB,3,1.65,0.794,1.35,0.088
9864156ed5c9eb6c,MarkerA,MarkerA,87,86.5,9.96,0.062,0.575
9864156ed5c9eb6c,MarkerB,MarkerC,30,27.3,3.1,0.001,0.999
c925e4e5eeb989b9,MarkerA,MarkerA,3,1.65,0.794,1.35,0.088
c925e4e5eeb989b9,MarkerB,MarkerC,87,86.5,9.96,0.062,0.575
c925e4e5eeb989b9,MarkerA,MarkerB,30,27.3,3.1,0.001,0.999
"""

DIFFERENTIAL_PROXIMITY_DATA = """marker_1,marker_2,reference,target,u_stat,p_value,auc,median_diff,tgt_median,ref_median,n_ref,n_tgt,p_adjusted
MarkerA,MarkerA,9864156ed5c9eb6c,c925e4e5eeb989b9,0.0,1.0,0.0,1.288,1.35,0.062,1,1,1.0
MarkerA,MarkerB,9864156ed5c9eb6c,c925e4e5eeb989b9,1.0,1.0,1.0,-1.35,0.001,1.35,1,1,1.0
MarkerB,MarkerC,9864156ed5c9eb6c,c925e4e5eeb989b9,0.0,1.0,0.0,0.061,0.062,0.001,1,1,1.0
"""


def test_proximity_analysis_jcs(pna_pxl_file: Path, pna_data_root, tmp_path):
    pna_pxl_dataset = PNAPixelDataset.from_files(pna_pxl_file)
    manager = AnalysisManager(
        [ProximityAnalysis(n_permutations=25, min_marker_count=0)],
        n_cores=3,
    )
    output_pxl_file = PxlFile.copy_pxl_file(
        PxlFile(pna_pxl_file), tmp_path / "proximity.pxl"
    )
    dataset = manager.execute(pna_pxl_dataset, output_pxl_file)
    proximity = dataset.proximity().to_df()

    assert "component" in proximity.columns

    assert "__index_level_0__" not in proximity.columns
    marker_types_count = (pna_pxl_dataset.adata().to_df() > 0).sum(axis=1)
    for comp, comp_proximity in proximity.groupby("component"):
        assert comp_proximity.shape[0] == (
            marker_types_count[comp] * (marker_types_count[comp] + 1) / 2
        )
    proximity = proximity.set_index(["component", "marker_1", "marker_2"], drop=True)

    expected_proximity = pd.read_csv(pna_data_root / "jcs_proximity.csv").set_index(
        ["component", "marker_1", "marker_2"], drop=True
    )
    expected_proximity = expected_proximity.astype({"join_count": "uint32"})

    assert_frame_equal(
        proximity, expected_proximity, check_like=True, check_dtype=False
    )


@pytest.mark.slow
def test_proximity_analysis_jcs_marker_count_filtering(
    pna_pxl_file: Path, pna_data_root, tmp_path
):
    pna_pxl_dataset = PNAPixelDataset.from_files(pna_pxl_file)
    min_marker_count = 10
    manager_unfiltered = AnalysisManager(
        [ProximityAnalysis(n_permutations=100, min_marker_count=0)],
        n_cores=3,
    )
    manager_filtered = AnalysisManager(
        [ProximityAnalysis(n_permutations=100, min_marker_count=min_marker_count)],
        n_cores=3,
    )
    output_pxl_file_unfiltered = PxlFile.copy_pxl_file(
        PxlFile(pna_pxl_file), tmp_path / "proximity_unfiltered.pxl"
    )
    output_pxl_file_filtered = PxlFile.copy_pxl_file(
        PxlFile(pna_pxl_file), tmp_path / "proximity_filtered.pxl"
    )
    dataset_unfiltered = manager_unfiltered.execute(
        pna_pxl_dataset, output_pxl_file_unfiltered
    )
    dataset_filtered = manager_filtered.execute(
        pna_pxl_dataset, output_pxl_file_filtered
    )

    proximity_unfiltered = dataset_unfiltered.proximity().to_df()
    proximity_filtered = dataset_filtered.proximity().to_df()

    assert proximity_filtered.shape[0] == (
        proximity_unfiltered.shape[0]
        - (proximity_unfiltered["min_count"] < min_marker_count).sum()
    )

    proximity_unfiltered = proximity_unfiltered.set_index(
        ["component", "marker_1", "marker_2"]
    ).sort_index()
    proximity_filtered = proximity_filtered.set_index(
        ["component", "marker_1", "marker_2"]
    ).sort_index()

    assert all(
        proximity_filtered["join_count"]
        == proximity_unfiltered.loc[
            proximity_unfiltered["min_count"] >= min_marker_count, "join_count"
        ]
    )


def test_calculate_differential_proximity():
    proximity = pd.read_csv(StringIO(PROXIMITY_DATA))
    dpa = calculate_differential_proximity(
        proximity,
        reference="9864156ed5c9eb6c",
        targets=["c925e4e5eeb989b9"],
        contrast_column="component",
    )
    dpa = dpa.set_index(["marker_1", "marker_2"], drop=True)
    expected_dpa = pd.read_csv(StringIO(DIFFERENTIAL_PROXIMITY_DATA)).set_index(
        ["marker_1", "marker_2"], drop=True
    )
    assert_frame_equal(dpa, expected_dpa, check_like=True, atol=1e-3)
