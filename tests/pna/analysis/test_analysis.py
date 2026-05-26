"""Analysis tests for pna data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import numpy as np
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


@pytest.mark.slow
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

    # Drop non numerical columns, as well as join_count_p since it varies too much between two different seeds
    drop_columns = ["join_count_p", "sample"]
    high_join_count_expected = expected_proximity.query("join_count_expected_mean > 1000").drop(columns=drop_columns)
    high_join_count_observed = proximity.loc[high_join_count_expected.index].drop(columns=drop_columns)

    # Check that the computed values are within relative tolerance. Some variables are very unstable.
    rtols = {
        'join_count': 1.e-5,
        'join_count_expected_mean': 0.02,
        'join_count_expected_sd': 0.75,
        'join_count_z': 15,
        'marker_1_count': 1.e-5,
        'marker_1_freq': 1.e-5,
        'marker_2_count': 1.e-5,
        'marker_2_freq': 1.e-5,
        'min_count': 1.e-5,
        'log2_ratio': 15,
    }
    tol_series = pd.Series(rtols)
    percent_diff = (
        abs(high_join_count_expected - high_join_count_observed)
        / (abs(high_join_count_expected) + 1e-9)
    )

    assert not (percent_diff > tol_series).any().any()


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


@pytest.mark.parametrize(
    "components,markers",
    [
        (None, None),
        (["d4074c845bb62800", "c3c393e9a17c1981"], None),
        (None, ["CD3e", "CD45RA", "CD44"]),
        (["d4074c845bb62800", "c3c393e9a17c1981"], ["CD3e", "CD45RA", "CD44"]),
    ],
)
def test_proximity_analysis_jcs_analytic(
    pna_pxl_file: Path, pna_data_root, components, markers
):
    pna_pxl_dataset = PNAPixelDataset.from_files(pna_pxl_file)
    proximity_obj = pna_pxl_dataset.filter(
        components=components, markers=markers
    ).proximity(calculate_from_edgelist=True)
    proximity = proximity_obj.to_df()
    proximity_len = proximity_obj.__len__()
    assert proximity_len == proximity.shape[0]
    assert len(proximity_obj) == proximity.shape[0]
    proximity = proximity.set_index(["component", "marker_1", "marker_2"], drop=True)

    expected_proximity = pd.read_csv(pna_data_root / "jcs_proximity.csv")
    if components is not None:
        expected_proximity = expected_proximity[
            expected_proximity["component"].isin(components)
        ]
    if markers is not None:
        expected_proximity = expected_proximity[
            expected_proximity["marker_1"].isin(markers)
            & expected_proximity["marker_2"].isin(markers)
        ]
    expected_proximity = expected_proximity.set_index(
        ["component", "marker_1", "marker_2"], drop=True
    )

    assert expected_proximity.shape == proximity.shape

    expected_proximity = expected_proximity.astype({"join_count": "uint32"})
    expected_proximity = expected_proximity[
        expected_proximity["join_count_expected_mean"] > 100
    ]
    proximity = proximity[proximity["join_count_expected_mean"] > 100]
    matching = proximity[["log2_ratio", "join_count_z", "join_count_p"]].join(
        expected_proximity[["log2_ratio", "join_count_z", "join_count_p"]],
        how="inner",
        rsuffix="_expected",
    )
    matching_corr = matching.corr()

    assert matching_corr.loc["log2_ratio", "log2_ratio_expected"] > 0.98
    assert matching_corr.loc["join_count_z", "join_count_z_expected"] > 0.98
    assert matching_corr.loc["join_count_p", "join_count_p_expected"] > 0.9
