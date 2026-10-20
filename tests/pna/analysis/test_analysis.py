"""Analysis tests for pna data.

Copyright © 2024 Pixelgen Technologies AB.
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal
from scipy.stats import mannwhitneyu

from pixelator.pna.analysis.analysis import ProximityAnalysis
from pixelator.pna.analysis.permute import edgelist_permutations
from pixelator.pna.analysis.proximity import (
    calculate_differential_proximity,
    jcs_with_analytical_stats_from_edgelist,
)
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
    """Run the proximity analysis and return its results.

    Args:
        pxl_file: Pxl file to analyze. It is copied, so it is left untouched.
        output_path: Where the copy holding the analysis results is written.
        n_cores: Number of cores used by the analysis manager.
        proximity_kwargs: Arguments for `ProximityAnalysis`.
    """
    dataset = PNAPixelDataset.from_files(pxl_file)
    manager = AnalysisManager(
        [ProximityAnalysis(**proximity_kwargs)],  # type: ignore[abstract]
        n_cores=n_cores,
    )
    output_pxl_file = PxlFile.copy_pxl_file(PxlFile(pxl_file), output_path)
    return manager.execute(dataset, output_pxl_file).proximity().to_df()


def test_proximity_analysis_jcs(synthetic_analysis_pxl_file: Path, tmp_path):
    """The proximity analysis gives the same scores as the analytic proximity.

    Both calculate the expected join counts analytically from the edgelist, so
    the scores stored by the analysis must match the ones calculated on the fly.

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

    assert (proximity["join_count"] == analytic["join_count"]).all()
    for column in [
        "join_count_expected_mean",
        "join_count_expected_sd",
        "join_count_z",
        "join_count_p",
        "log2_ratio",
    ]:
        np.testing.assert_allclose(proximity[column], analytic[column], err_msg=column)


def _random_edgelist(seed: int = 7) -> pl.DataFrame:
    """Create a random bipartite edgelist with uneven node degrees."""
    rng = np.random.default_rng(seed)
    n_edges, n_umi1, n_umi2 = 3000, 300, 400
    # Skewed node weights give a wide degree distribution
    umi1 = rng.choice(n_umi1, size=n_edges, p=rng.dirichlet(np.full(n_umi1, 0.5)))
    umi2 = rng.choice(n_umi2, size=n_edges, p=rng.dirichlet(np.full(n_umi2, 0.5)))
    edgelist = pl.DataFrame(
        {"umi1": umi1.astype(np.uint64), "umi2": umi2.astype(np.uint64)}
    ).unique()
    markers = np.array(["A", "B", "C", "D"])
    marker_p = [0.5, 0.25, 0.15, 0.1]
    umi1_markers = rng.choice(markers, size=n_umi1, p=marker_p)
    umi2_markers = rng.choice(markers, size=n_umi2, p=marker_p)
    return edgelist.with_columns(
        marker_1=pl.Series(umi1_markers[edgelist["umi1"].to_numpy()]),
        marker_2=pl.Series(umi2_markers[edgelist["umi2"].to_numpy()]),
    )


def test_analytical_proximity_stats_match_permutations():
    """Verify that the analytical expectations match the marker permutation null."""
    edgelist = _random_edgelist()
    result = jcs_with_analytical_stats_from_edgelist(edgelist).set_index(
        ["marker_1", "marker_2"]
    )

    n_permutations = 2000
    permuted_join_counts = pd.concat(
        perm.select(
            marker_1=pl.min_horizontal("marker_1", "marker_2"),
            marker_2=pl.max_horizontal("marker_1", "marker_2"),
        )
        .group_by("marker_1", "marker_2")
        .len()
        .with_columns(perm_idx=pl.lit(idx))
        .to_pandas()
        for idx, perm in enumerate(
            edgelist_permutations(edgelist, n=n_permutations, random_seed=1)
        )
    )
    permuted_join_counts = (
        permuted_join_counts.pivot_table(
            index=["marker_1", "marker_2"], columns="perm_idx", values="len"
        )
        .reindex(columns=range(n_permutations))
        .fillna(0)
    )
    permuted_mean = permuted_join_counts.mean(axis=1)
    permuted_sd = permuted_join_counts.std(axis=1)

    assert set(permuted_mean.index) == set(result.index)
    np.testing.assert_allclose(
        result["join_count_expected_mean"],
        permuted_mean.loc[result.index],
        rtol=0.02,
    )
    # The Monte Carlo standard error of the permutation sd is ~1.6% with 2000 permutations
    np.testing.assert_allclose(
        result["join_count_expected_sd"],
        permuted_sd.loc[result.index],
        rtol=0.08,
    )
    np.testing.assert_allclose(
        result["join_count_z"],
        (result["join_count"] - result["join_count_expected_mean"])
        / np.maximum(result["join_count_expected_sd"], 1.0),
    )


def test_analytical_proximity_stats_output_format():
    """Verify the output format of the analytical proximity statistics."""
    edgelist = _random_edgelist()
    result = jcs_with_analytical_stats_from_edgelist(edgelist)

    assert list(result.columns) == [
        "marker_1",
        "marker_2",
        "join_count",
        "join_count_expected_mean",
        "join_count_expected_sd",
        "join_count_z",
        "join_count_p",
    ]
    assert result["join_count"].dtype == np.uint32
    assert (result["marker_1"] <= result["marker_2"]).all()
    assert result.shape[0] == 4 * 5 / 2
    assert result["join_count"].sum() == edgelist.height


def test_proximity_analysis_jcs_marker_count_filtering(
    synthetic_analysis_pxl_file: Path, tmp_path
):
    """`min_marker_count` removes exactly the marker pairs with too few markers.

    Args:
        synthetic_analysis_pxl_file: Synthetic analysis pxl file.
        tmp_path: tmp path.
    """
    min_marker_count = 10
    unfiltered = _run_proximity_analysis(
        synthetic_analysis_pxl_file,
        tmp_path / "proximity_unfiltered.pxl",
        n_cores=1,
        min_marker_count=0,
    )
    filtered = _run_proximity_analysis(
        synthetic_analysis_pxl_file,
        tmp_path / "proximity_filtered.pxl",
        n_cores=1,
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


def _stratum_block(sample: str, ref_value: float, tgt_value: float) -> pd.DataFrame:
    components = [f"{sample}_a{i}" for i in range(10)] + [
        f"{sample}_b{i}" for i in range(10)
    ]
    return pd.DataFrame(
        {
            "component": components,
            "sample": [sample] * 20,
            "cell_type": ["A"] * 10 + ["B"] * 10,
            "marker_1": ["M1"] * 20,
            "marker_2": ["M1"] * 20,
            "log2_ratio": [ref_value] * 10 + [tgt_value] * 10,
        }
    )


def test_calculate_differential_proximity_group_vars_adjusts_within_stratum():
    """Each grouping combination is tested and adjusted on its own."""
    proximity = pd.concat(
        [_stratum_block("s1", 1.0, 5.0), _stratum_block("s2", 2.0, 9.0)],
        ignore_index=True,
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="cell_type",
        reference="A",
        targets=["B"],
        group_vars="sample",
    )
    assert list(result["sample"]) == ["s1", "s2"]
    assert result["n_ref"].tolist() == [10, 10]
    assert result["n_tgt"].tolist() == [10, 10]
    assert result["p_adjusted"].tolist() == pytest.approx(result["p_value"].tolist())


def test_calculate_differential_proximity_group_vars_skips_a_small_stratum():
    """A grouping combination with too few cells is skipped on its own."""
    small = _stratum_block("s2", 1.0, 5.0).iloc[:9]
    proximity = pd.concat([_stratum_block("s1", 1.0, 5.0), small], ignore_index=True)
    with pytest.warns(UserWarning, match=r"sample=s2"):
        result = calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            group_vars="sample",
        )
    assert result["sample"].tolist() == ["s1"]
    assert result.loc[0, "n_ref"] == 10


def test_calculate_differential_proximity_group_vars_rejects_missing_values():
    """A missing grouping value is an error, not a dropped stratum."""
    proximity = _stratum_block("s1", 1.0, 5.0)
    proximity.loc[0, "sample"] = pd.NA
    with pytest.raises(ValueError, match="missing values"):
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            group_vars="sample",
        )


def test_calculate_differential_proximity_rejects_shared_missing_group_label():
    """The same missing grouping label in both tables is not a disagreement."""
    proximity = _stratum_block("s1", 1.0, 5.0)
    component = proximity.loc[0, "component"]
    proximity.loc[proximity["component"] == component, "sample"] = pd.NA
    cell_groups = proximity[["component", "cell_type", "sample"]].drop_duplicates()
    with pytest.raises(ValueError, match="missing values") as error:
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            group_vars="sample",
            cell_groups=cell_groups,
        )
    assert "disagree" not in str(error.value)


def test_calculate_differential_proximity_rejects_shared_missing_contrast_label():
    """The same missing contrast label in both tables is not a disagreement."""
    proximity = _stratum_block("s1", 1.0, 5.0)
    component = proximity.loc[0, "component"]
    proximity.loc[proximity["component"] == component, "cell_type"] = pd.NA
    cell_groups = proximity[["component", "cell_type"]].drop_duplicates()
    with pytest.raises(ValueError, match="missing values") as error:
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            cell_groups=cell_groups,
        )
    assert "disagree" not in str(error.value)


def test_calculate_differential_proximity_rejects_missing_contrast_before_targets():
    """An omitted target list does not turn a missing contrast label into a target."""
    proximity = pd.DataFrame(
        {
            "component": ["a", "b"],
            "cell_type": ["A", "B"],
            "marker_1": ["M1", "M1"],
            "marker_2": ["M1", "M1"],
            "log2_ratio": [1.0, 2.0],
        }
    )
    cell_groups = pd.DataFrame({"component": ["a", "b"], "cell_type": ["A", pd.NA]})
    with pytest.raises(ValueError, match="missing values"):
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            cell_groups=cell_groups,
        )


def test_calculate_differential_proximity_rejects_missing_contrast_without_component():
    """Without a component id, a missing contrast label is not its own target."""
    proximity = pd.DataFrame(
        {
            "sample": ["A", np.nan],
            "marker_1": ["M1", "M1"],
            "marker_2": ["M1", "M1"],
            "log2_ratio": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="missing values"):
        calculate_differential_proximity(
            proximity,
            contrast_column="sample",
            reference="A",
        )


def test_calculate_differential_proximity_group_vars_rejects_two_values():
    """A component cannot belong to two values of a grouping column."""
    proximity = _stratum_block("s1", 1.0, 5.0)
    extra = proximity.iloc[[0]].copy()
    extra["sample"] = "s2"
    extra["marker_2"] = "M2"
    proximity = pd.concat([proximity, extra], ignore_index=True)
    with pytest.raises(ValueError, match="more than one value"):
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            group_vars="sample",
        )


def test_calculate_differential_proximity_min_n_obs_allows_group_with_no_rows():
    """A cell_groups-only group counts as zero detections under min_n_obs."""
    proximity = pd.DataFrame(
        {
            "component": [f"a{i}" for i in range(10)],
            "cell_type": ["A"] * 10,
            "marker_1": ["M1"] * 10,
            "marker_2": ["M1"] * 10,
            "log2_ratio": [1.0] * 10,
        }
    )
    cell_groups = pd.DataFrame(
        {
            "component": [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)],
            "cell_type": ["A"] * 10 + ["B"] * 10,
        }
    )
    result = calculate_differential_proximity(
        proximity,
        contrast_column="cell_type",
        reference="A",
        targets=["B"],
        cell_groups=cell_groups,
        min_n_obs=1,
    )
    assert result.empty
    assert "p_adjusted" in result.columns


def test_calculate_differential_proximity_group_vars_rejects_the_contrast_column():
    """The contrast column is not a grouping variable."""
    proximity = _stratum_block("s1", 1.0, 5.0)
    with pytest.raises(ValueError, match="cannot be one of"):
        calculate_differential_proximity(
            proximity,
            contrast_column="cell_type",
            reference="A",
            targets=["B"],
            group_vars="cell_type",
        )


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
