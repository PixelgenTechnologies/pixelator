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


PROXIMITY_KEY = ["component", "marker_1", "marker_2"]


def _run_proximity_analysis(
    pxl_file: Path, output_path: Path, n_cores: int, **proximity_kwargs
) -> pd.DataFrame:
    """Run the permutation based proximity analysis and return its results.

    Args:
        pxl_file: Pxl file to analyze. It is copied, so it is left untouched.
        output_path: Where the copy holding the analysis results is written.
        n_cores: Number of cores used by the analysis manager.
        proximity_kwargs: Arguments for `ProximityAnalysis`.
    """
    dataset = PNAPixelDataset.from_files(pxl_file)
    manager = AnalysisManager([ProximityAnalysis(**proximity_kwargs)], n_cores=n_cores)
    output_pxl_file = PxlFile.copy_pxl_file(PxlFile(pxl_file), output_path)
    return manager.execute(dataset, output_pxl_file).proximity().to_df()


def test_proximity_analysis_jcs(synthetic_analysis_pxl_file: Path, tmp_path):
    """The permutation based proximity analysis agrees with the analytic proximity.

    The analytic proximity (computed directly from the edgelist) is the
    reference: the permutation estimates must be close to it.

    Args:
        synthetic_analysis_pxl_file: Synthetic analysis pxl file.
        tmp_path: tmp path.
    """
    # Two cores on purpose: this is the one test that runs the analysis in a
    # worker pool.
    proximity = _run_proximity_analysis(
        synthetic_analysis_pxl_file,
        tmp_path / "proximity.pxl",
        n_cores=2,
        n_permutations=25,
        min_marker_count=0,
    )
    dataset = PNAPixelDataset.from_files(synthetic_analysis_pxl_file)
    analytic = dataset.proximity(calculate_from_edgelist=True).to_df()

    # Structure: no leftover index column, and every component has one row per
    # unordered pair of the markers it contains (including a marker with itself).
    assert "component" in proximity.columns
    assert "__index_level_0__" not in proximity.columns
    n_markers = (dataset.adata().to_df() > 0).sum(axis=1)
    for component, rows in proximity.groupby("component"):
        assert len(rows) == n_markers[component] * (n_markers[component] + 1) / 2

    # Line the two results up row by row.
    proximity = proximity.set_index(PROXIMITY_KEY)
    analytic = analytic.set_index(PROXIMITY_KEY)
    assert set(analytic.index) == set(proximity.index)
    analytic = analytic.loc[proximity.index]

    # The observed join counts are exact, they do not depend on the permutations.
    assert (proximity["join_count"] == analytic["join_count"]).all()

    # The permutation estimates are noisy for marker pairs with few expected
    # joins, so only compare the pairs with enough of them.
    enough_joins = analytic["join_count_expected_mean"] > 5
    assert enough_joins.sum() > 100
    estimated = proximity[enough_joins]
    reference = analytic[enough_joins]
    assert (
        estimated["join_count_expected_mean"].corr(
            reference["join_count_expected_mean"]
        )
        > 0.99
    )
    assert estimated["log2_ratio"].corr(reference["log2_ratio"]) > 0.9
    assert estimated["join_count_z"].corr(reference["join_count_z"]) > 0.9


def test_proximity_analysis_jcs_marker_count_filtering(
    synthetic_analysis_pxl_file: Path, tmp_path
):
    """`min_marker_count` removes exactly the marker pairs with too few markers.

    Args:
        synthetic_analysis_pxl_file: Synthetic analysis pxl file.
        tmp_path: tmp path.
    """
    min_marker_count = 10
    # Only the observed join counts are compared, which do not depend on the
    # permutations, so a handful of them is enough.
    unfiltered = _run_proximity_analysis(
        synthetic_analysis_pxl_file,
        tmp_path / "proximity_unfiltered.pxl",
        n_cores=1,
        n_permutations=10,
        min_marker_count=0,
    )
    filtered = _run_proximity_analysis(
        synthetic_analysis_pxl_file,
        tmp_path / "proximity_filtered.pxl",
        n_cores=1,
        n_permutations=10,
        min_marker_count=min_marker_count,
    )

    # Without this the test would pass even if nothing could be filtered out.
    too_few_markers = unfiltered["min_count"] < min_marker_count
    assert too_few_markers.any()

    # The filtered result is the unfiltered one minus the pairs with too few
    # markers, with unchanged join counts.
    assert len(filtered) == len(unfiltered) - too_few_markers.sum()
    kept = unfiltered[~too_few_markers].set_index(PROXIMITY_KEY).sort_index()
    filtered = filtered.set_index(PROXIMITY_KEY).sort_index()
    assert (filtered["join_count"] == kept["join_count"]).all()


def test_calculate_differential_proximity():
    """Verify calculate differential proximity."""
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
    """Verify proximity analysis jcs analytic.

    Args:
        pna_data_root: pna data root.
        components: components.
        markers: markers.
        pna_pxl_file: Pna pxl file.
    """
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
