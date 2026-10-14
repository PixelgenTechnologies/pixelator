"""Analysis tests for pna data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import mannwhitneyu

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
    """Verify proximity analysis jcs.

    Args:
        pna_data_root: pna data root.
        tmp_path: tmp path.
        pna_pxl_file: Pna pxl file.
    """
    pna_pxl_dataset = PNAPixelDataset.from_files(pna_pxl_file)
    manager = AnalysisManager(
        [ProximityAnalysis(n_permutations=25, min_marker_count=0)],  # type: ignore[abstract]
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
    high_join_count_expected = expected_proximity.query(
        "join_count_expected_mean > 1000"
    ).drop(columns=drop_columns)
    high_join_count_observed = proximity.loc[high_join_count_expected.index].drop(
        columns=drop_columns
    )

    # Check that the computed values are within relative tolerance. Some variables are very unstable.
    rtols = {
        "join_count": 1.0e-5,
        "join_count_expected_mean": 0.02,
        "join_count_expected_sd": 0.75,
        "join_count_z": 15,
        "marker_1_count": 1.0e-5,
        "marker_1_freq": 1.0e-5,
        "marker_2_count": 1.0e-5,
        "marker_2_freq": 1.0e-5,
        "min_count": 1.0e-5,
        "log2_ratio": 15,
    }
    tol_series = pd.Series(rtols)
    percent_diff = abs(high_join_count_expected - high_join_count_observed) / (
        abs(high_join_count_expected) + 1e-9
    )

    assert not (percent_diff > tol_series).any().any()


@pytest.mark.slow
def test_proximity_analysis_jcs_marker_count_filtering(
    pna_pxl_file: Path, pna_data_root, tmp_path
):
    """Verify proximity analysis jcs marker count filtering.

    Args:
        pna_data_root: pna data root.
        tmp_path: tmp path.
        pna_pxl_file: Pna pxl file.
    """
    pna_pxl_dataset = PNAPixelDataset.from_files(pna_pxl_file)
    min_marker_count = 10
    manager_unfiltered = AnalysisManager(
        [ProximityAnalysis(n_permutations=100, min_marker_count=0)],  # type: ignore[abstract]
        n_cores=3,
    )
    manager_filtered = AnalysisManager(
        [ProximityAnalysis(n_permutations=100, min_marker_count=min_marker_count)],  # type: ignore[abstract]
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


# Five-cell PBMC file (pna_pxl_dataset). Obs order matches the file's component order.
_FIVE_CELL_TYPES = ("CD16+ Mono", "pDC", "CD4T", "CD4T", "CD4T")


def test_calculate_differential_proximity_five_cell_pbmc(pna_pxl_dataset):
    """CD4T versus pDC on the five-cell PBMC proximity table, copied ten times.

    One CD4T component has no CD56 rows. ``cell_groups`` still lists that cell,
    and each of the five cells is copied ten times, so the CD4T median is
    taken over 30 cells and the pDC median over 10.
    """
    components = list(pna_pxl_dataset.adata().obs.index)
    proximity = pna_pxl_dataset.proximity(add_marker_counts=False).to_df()
    proximity["cell_type"] = proximity["component"].map(
        dict(zip(components, _FIVE_CELL_TYPES))
    )
    pairs = proximity["marker_1"].eq("CD56") & proximity["marker_2"].isin(
        ["HLA-ABC", "HLA-DR-DP-DQ"]
    )
    proximity = proximity.loc[pairs].copy()
    cell_groups = pd.DataFrame(
        {"component": components, "cell_type": list(_FIVE_CELL_TYPES)}
    )
    copies = []
    group_copies = []
    for copy in "ABCDEFGHIJ":
        copied = proximity.copy()
        copied["component"] = copy + "_" + copied["component"]
        copies.append(copied)
        grouped = cell_groups.copy()
        grouped["component"] = copy + "_" + grouped["component"]
        group_copies.append(grouped)
    proximity = pd.concat(copies, ignore_index=True)
    cell_groups = pd.concat(group_copies, ignore_index=True)

    result = calculate_differential_proximity(
        proximity,
        contrast_column="cell_type",
        reference="pDC",
        targets=["CD4T"],
        diff_threshold=1,
        cell_groups=cell_groups,
    ).set_index(["marker_1", "marker_2"])

    hla_abc = result.loc[("CD56", "HLA-ABC")]
    assert hla_abc["n_ref"] == 10
    assert hla_abc["n_tgt"] == 30
    assert hla_abc["pct_ref"] == 1
    assert hla_abc["pct_tgt"] == pytest.approx(0.333)
    assert hla_abc["median_diff"] == pytest.approx(1.13093086982645)
    assert hla_abc["p_value"] == pytest.approx(0.0924377451134652)

    hla_dr = result.loc[("CD56", "HLA-DR-DP-DQ")]
    assert hla_dr["n_ref"] == 10
    assert hla_dr["n_tgt"] == 30
    assert hla_dr["pct_ref"] == 1
    assert hla_dr["pct_tgt"] == pytest.approx(0.333)
    assert hla_dr["median_diff"] == pytest.approx(2.06004738366994)
    assert hla_dr["p_value"] == pytest.approx(3.73436610215361e-07)


def test_calculate_differential_proximity_five_cell_pbmc_adjusts_all_pairs(
    pna_pxl_dataset,
):
    """Bonferroni for the five-cell PBMC comparison uses every pair that passes.

    The two CD56 pairs above are corrected together with the rest of the
    table. With ``diff_threshold=1`` that family has 931 pairs, and the
    adjusted p-value of CD56/HLA-DR-DP-DQ is the raw p-value times 931.
    """
    components = list(pna_pxl_dataset.adata().obs.index)
    proximity = pna_pxl_dataset.proximity(add_marker_counts=False).to_df()
    proximity["cell_type"] = proximity["component"].map(
        dict(zip(components, _FIVE_CELL_TYPES))
    )
    cell_groups = pd.DataFrame(
        {"component": components, "cell_type": list(_FIVE_CELL_TYPES)}
    )
    copies = []
    group_copies = []
    for copy in "ABCDEFGHIJ":
        copied = proximity.copy()
        copied["component"] = copy + "_" + copied["component"].astype(str)
        copies.append(copied)
        grouped = cell_groups.copy()
        grouped["component"] = copy + "_" + grouped["component"].astype(str)
        group_copies.append(grouped)
    proximity = pd.concat(copies, ignore_index=True)
    cell_groups = pd.concat(group_copies, ignore_index=True)

    result = calculate_differential_proximity(
        proximity,
        contrast_column="cell_type",
        reference="pDC",
        targets=["CD4T"],
        diff_threshold=1,
        cell_groups=cell_groups,
    ).set_index(["marker_1", "marker_2"])

    assert len(result) == 931
    hla_dr = result.loc[("CD56", "HLA-DR-DP-DQ")]
    assert hla_dr["p_value"] == pytest.approx(3.73436610215361e-07)
    assert hla_dr["p_adjusted"] == pytest.approx(0.000347669484110502)


def test_calculate_differential_proximity():
    """Verify calculate differential proximity."""
    proximity = pd.read_csv(StringIO(PROXIMITY_DATA))
    dpa = calculate_differential_proximity(
        proximity,
        reference="9864156ed5c9eb6c",
        targets=["c925e4e5eeb989b9"],
        contrast_column="component",
        metric="join_count_z",
        min_cells_per_group=1,
    )
    dpa = dpa.set_index(["marker_1", "marker_2"], drop=True)
    expected_dpa = pd.read_csv(StringIO(DIFFERENTIAL_PROXIMITY_DATA)).set_index(
        ["marker_1", "marker_2"], drop=True
    )
    assert_frame_equal(
        dpa[expected_dpa.columns], expected_dpa, check_like=True, atol=1e-3
    )


def test_calculate_differential_proximity_zero_pads_missing_components():
    """Missing component/pair observations of the tested metric count as 0."""
    # Sample A is c1, c2, c3. Sample B is c4, c5.
    # M1/M1 is missing in c3 and c5. M1/M2 is observed only in c1.
    # M2/M2 exists so c3 and c5 are part of the component universe.
    proximity = pd.DataFrame(
        {
            "component": ["c1", "c2", "c3", "c4", "c5", "c1"],
            "sample": ["A", "A", "A", "B", "B", "A"],
            "marker_1": ["M1", "M1", "M2", "M1", "M2", "M1"],
            "marker_2": ["M1", "M1", "M2", "M1", "M2", "M2"],
            "join_count_z": [1.0, 3.0, -1.0, 2.0, -2.0, 4.0],
        }
    )

    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        metric="join_count_z",
        min_cells_per_group=1,
        diff_threshold=0,
    ).set_index(["marker_1", "marker_2"])

    ref_m1 = np.array([1.0, 3.0, 0.0])
    tgt_m1 = np.array([2.0, 0.0])
    u_m1, p_m1 = mannwhitneyu(
        ref_m1,
        tgt_m1,
        alternative="two-sided",
        use_continuity=True,
        method="asymptotic",
    )
    m1 = result.loc[("M1", "M1")]
    assert m1["n_ref"] == 3
    assert m1["n_tgt"] == 2
    assert m1["ref_median"] == np.median(ref_m1)
    assert m1["tgt_median"] == np.median(tgt_m1)
    assert m1["median_diff"] == np.median(tgt_m1) - np.median(ref_m1)
    assert m1["u_stat"] == u_m1
    assert m1["p_value"] == pytest.approx(p_m1)
    assert m1["auc"] == u_m1 / (len(ref_m1) * len(tgt_m1))

    ref_m2 = np.array([4.0, 0.0, 0.0])
    tgt_m2 = np.array([0.0, 0.0])
    u_m2, p_m2 = mannwhitneyu(
        ref_m2,
        tgt_m2,
        alternative="two-sided",
        use_continuity=True,
        method="asymptotic",
    )
    m2 = result.loc[("M1", "M2")]
    assert m2["n_ref"] == 3
    assert m2["n_tgt"] == 2
    assert m2["ref_median"] == 0.0
    assert m2["tgt_median"] == 0.0
    assert m2["u_stat"] == u_m2
    assert m2["p_value"] == pytest.approx(p_m2)


def test_calculate_differential_proximity_without_component_uses_observed_rows():
    """Without a component id, pairs missing from a group are left out."""
    proximity = pd.DataFrame(
        {
            "sample": ["A", "A", "B"],
            "marker_1": ["M1", "M1", "M1"],
            "marker_2": ["M1", "M2", "M1"],
            "join_count_z": [1.0, 4.0, 2.0],
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        metric="join_count_z",
    ).set_index(["marker_1", "marker_2"])

    assert list(result.index) == [("M1", "M1")]
    assert result.loc[("M1", "M1"), "n_ref"] == 1
    assert result.loc[("M1", "M1"), "n_tgt"] == 1
    assert result.loc[("M1", "M1"), "ref_median"] == 1.0
    assert result.loc[("M1", "M1"), "tgt_median"] == 2.0


def test_calculate_differential_proximity_skips_groups_with_fewer_than_10_cells():
    """A comparison is skipped when either group has fewer than 10 cells."""
    components = [f"c{i}" for i in range(19)]
    proximity = pd.DataFrame(
        {
            "component": components,
            "sample": ["A"] * 9 + ["B"] * 10,
            "marker_1": ["M1"] * 19,
            "marker_2": ["M1"] * 19,
            "log2_ratio": range(19),
        }
    )
    with pytest.warns(UserWarning, match="fewer than 10 cells"):
        skipped = calculate_differential_proximity(
            proximity,
            contrast_column="sample",
            reference="A",
            targets=["B"],
        )
    assert skipped.empty
    assert "p_adjusted" in skipped.columns
    assert "median_diff" in skipped.columns

    kept = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        min_cells_per_group=9,
    )
    assert len(kept) == 1
    assert kept.loc[0, "n_ref"] == 9
    assert kept.loc[0, "n_tgt"] == 10


def test_calculate_differential_proximity_filters_before_adjusting_pvalues():
    """Pairs below the median-difference threshold are left out of the correction."""
    components = [f"c{i}" for i in range(20)]
    proximity = pd.DataFrame(
        {
            "component": components * 2,
            "sample": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
            "marker_1": ["M1"] * 20 + ["M2"] * 20,
            "marker_2": ["M1"] * 20 + ["M2"] * 20,
            "log2_ratio": [1.0] * 10 + [3.0] * 10 + [1.0] * 20,
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
    ).set_index(["marker_1", "marker_2"])

    assert list(result.index) == [("M1", "M1")]
    assert result.loc[("M1", "M1"), "p_adjusted"] == result.loc[("M1", "M1"), "p_value"]


def test_calculate_differential_proximity_pads_cells_absent_from_proximity():
    """A cell listed in cell_groups but missing from the proximity table is a 0."""
    proximity = pd.DataFrame(
        {
            "component": [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(9)],
            "sample": ["A"] * 10 + ["B"] * 9,
            "marker_1": ["M1"] * 19,
            "marker_2": ["M1"] * 19,
            "log2_ratio": [1.0] * 10 + [0.0] * 9,
        }
    )
    cell_groups = pd.DataFrame(
        {
            "component": [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)],
            "sample": ["A"] * 10 + ["B"] * 10,
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        cell_groups=cell_groups,
        diff_threshold=0,
    )
    assert result.loc[0, "n_ref"] == 10
    assert result.loc[0, "n_tgt"] == 10
    assert result.loc[0, "tgt_median"] == 0.0
    assert result.loc[0, "pct_tgt"] == 0.0


def test_calculate_differential_proximity_rejects_two_labels_for_one_component():
    """One component cannot belong to two groups in the proximity table."""
    proximity = pd.DataFrame(
        {
            "component": ["a", "a"],
            "sample": ["A", "B"],
            "marker_1": ["M1", "M1"],
            "marker_2": ["M1", "M2"],
            "log2_ratio": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="more than one"):
        calculate_differential_proximity(
            proximity,
            contrast_column="sample",
            reference="A",
            targets=["B"],
            min_cells_per_group=1,
        )


def test_calculate_differential_proximity_rejects_conflicting_cell_labels():
    """A component cannot move groups between the two tables."""
    proximity = pd.DataFrame(
        {
            "component": ["a", "b"],
            "sample": ["A", "B"],
            "marker_1": ["M1", "M1"],
            "marker_2": ["M1", "M1"],
            "log2_ratio": [1.0, 2.0],
        }
    )
    cell_groups = pd.DataFrame({"component": ["a", "b"], "sample": ["B", "A"]})
    with pytest.raises(ValueError, match="disagree"):
        calculate_differential_proximity(
            proximity,
            contrast_column="sample",
            reference="A",
            targets=["B"],
            cell_groups=cell_groups,
        )


def test_calculate_differential_proximity_rejects_conflicting_scores():
    """Two different scores for one component and pair are an error."""
    proximity = pd.DataFrame(
        {
            "component": ["a", "a", "b"],
            "sample": ["A", "A", "B"],
            "marker_1": ["M1", "M1", "M1"],
            "marker_2": ["M1", "M1", "M1"],
            "log2_ratio": [1.0, 4.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="conflicting"):
        calculate_differential_proximity(
            proximity,
            contrast_column="sample",
            reference="A",
            targets=["B"],
            min_cells_per_group=1,
        )


def test_calculate_differential_proximity_collapses_identical_duplicate_rows():
    """Repeated copies of the same score do not count as extra cells."""
    components = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
    proximity = pd.DataFrame(
        {
            "component": components,
            "sample": ["A"] * 10 + ["B"] * 10,
            "marker_1": ["M1"] * 20,
            "marker_2": ["M1"] * 20,
            "log2_ratio": [1.0] * 10 + [3.0] * 10,
        }
    )
    proximity = pd.concat([proximity, proximity.iloc[[0]]], ignore_index=True)
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
    )
    assert result.loc[0, "n_ref"] == 10
    assert result.loc[0, "n_tgt"] == 10


def test_calculate_differential_proximity_tied_groups_have_p_value_one():
    """A fully tied comparison is reported as p = 1, not a missing p-value."""
    components = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
    proximity = pd.DataFrame(
        {
            "component": components * 2,
            "sample": (["A"] * 10 + ["B"] * 10) * 2,
            "marker_1": ["M1"] * 20 + ["M2"] * 20,
            "marker_2": ["M1"] * 20 + ["M2"] * 20,
            "log2_ratio": [0.0] * 20 + [1.0] * 10 + [4.0] * 10,
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        diff_threshold=0,
    ).set_index(["marker_1", "marker_2"])
    assert result.loc[("M1", "M1"), "p_value"] == 1.0
    assert result.loc[("M1", "M1"), "p_adjusted"] == 1.0
    assert result.loc[("M2", "M2"), "p_adjusted"] == pytest.approx(
        2 * result.loc[("M2", "M2"), "p_value"]
    )


def test_calculate_differential_proximity_uses_normal_approximation():
    """Small samples without ties use the normal approximation, not the exact test."""
    ref = np.arange(1, 6, dtype=float)
    tgt = np.arange(6, 11, dtype=float)
    proximity = pd.DataFrame(
        {
            "component": [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)],
            "sample": ["A"] * 5 + ["B"] * 5,
            "marker_1": ["M1"] * 10,
            "marker_2": ["M1"] * 10,
            "log2_ratio": np.concatenate([ref, tgt]),
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
        min_cells_per_group=1,
    )
    asymptotic = mannwhitneyu(
        ref, tgt, alternative="two-sided", use_continuity=True, method="asymptotic"
    ).pvalue
    exact = mannwhitneyu(ref, tgt, alternative="two-sided", method="exact").pvalue
    assert result.loc[0, "p_value"] == pytest.approx(asymptotic)
    assert result.loc[0, "p_value"] != pytest.approx(exact)


def test_calculate_differential_proximity_threshold_keeps_result_columns():
    """A comparison whose pairs are all filtered still has the result columns."""
    components = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)]
    proximity = pd.DataFrame(
        {
            "component": components,
            "sample": ["A"] * 10 + ["B"] * 10,
            "marker_1": ["M1"] * 20,
            "marker_2": ["M1"] * 20,
            "log2_ratio": [1.0] * 20,
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="sample",
        reference="A",
        targets=["B"],
    )
    assert result.empty
    assert "p_adjusted" in result.columns
    assert "median_diff" in result.columns


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
