"""Tests for the sample calling module.

Copyright © 2025 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pytest

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna import read
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset.io import PixelFileWriter
from pixelator.pna.sample_calling import (
    collect_hash_info,
    create_final_report,
    sample_calling,
    warn_if_undetermined_has_high_confidence,
)
from pixelator.pna.sample_calling.hash_antibodies import HashedAntibodyMapping
from pixelator.pna.sample_calling.sample_calling import (
    _add_original_hash_counts_to_obs,
    _collect_nodes_to_remove,
)


def test_add_original_hash_counts_includes_all_panel_antibodies():
    """All panel hashing antibodies get original_hash_counts_* in obs, including those not in adata.

    When the panel has more hashing antibodies than the samplesheet (or some antibodies
    have no counts), obs should still have a column for each; missing ones are 0.
    """

    # adata has only h1, h2 in var (e.g. panel has h1, h2, h3)
    adata = anndata.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 0.0]]),
        obs=dict(component=["c1", "c2"]),
        var=dict(gene_name=["h1", "h2"]),
    )
    adata.var.index = ["h1", "h2"]
    adata.obs.set_index("component", inplace=True)
    antibodies_for_obs = ["h1", "h2", "h3"]

    _add_original_hash_counts_to_obs(adata, antibodies_for_obs)

    assert "original_hash_counts_h1" in adata.obs.columns
    assert "original_hash_counts_h2" in adata.obs.columns
    assert "original_hash_counts_h3" in adata.obs.columns
    np.testing.assert_array_equal(adata.obs["original_hash_counts_h1"], [1.0, 3.0])
    np.testing.assert_array_equal(adata.obs["original_hash_counts_h2"], [2.0, 0.0])
    np.testing.assert_array_equal(adata.obs["original_hash_counts_h3"], [0.0, 0.0])


def _samplesheet_and_hashing_for_three_samples():
    """Samplesheet and full hashing list for PBMC/Raji/Jurkat (hash 1/2/3).

    Use for tests that need HashedAntibodyMapping.
    """
    samplesheet = pl.DataFrame(
        {
            "pool": ["test_pool", "test_pool", "test_pool"],
            "sample": ["PBMC", "Raji", "Jurkat"],
            "hash_index": [1, 2, 3],
        }
    )
    all_hashing = {"B2M-1", "B2M-2", "B2M-3", "CD29-1", "CD29-2", "CD29-3"}
    return samplesheet, all_hashing


def test_hashed_antibody_mapping_requires_samplesheet_and_all_hashing_antibodies():
    """HashedAntibodyMapping is created only via from_samplesheet with both samplesheet and all_hashing_antibodies."""
    samplesheet, all_hashing = _samplesheet_and_hashing_for_three_samples()
    mapping = HashedAntibodyMapping.from_samplesheet(
        samplesheet,
        all_hashing_antibodies=all_hashing,
        pool_name="test_pool",
    )
    assert mapping["PBMC"] == ["B2M-1", "CD29-1"]
    assert mapping["Raji"] == ["B2M-2", "CD29-2"]
    assert mapping["Jurkat"] == ["B2M-3", "CD29-3"]
    assert mapping.hashing_antibodies == all_hashing
    # Only from_samplesheet is supported;
    # from_dict is removed so both samplesheet and all_hashing_antibodies are always required
    assert not hasattr(HashedAntibodyMapping, "from_dict"), (
        "from_dict must be removed; "
        + "use from_samplesheet(samplesheet, all_hashing_antibodies, pool_name)"
    )


def test_hashed_antibody_mapping_raises_when_antibody_in_multiple_samples():
    """HashedAntibodyMapping raises ValueError when the same antibody is assigned to more than one sample."""
    with pytest.raises(
        ValueError, match="Antibody hash 'h1' is assigned to multiple samples"
    ):
        HashedAntibodyMapping(
            mapping={"A": ["h1"], "B": ["h1"]},
            all_hashing_antibodies=["h1"],
        )


def test_from_samplesheet_raises_when_hash_index_column_missing():
    """from_samplesheet requires a column named exactly 'hash_index'."""
    df = pl.DataFrame(
        {
            "pool": ["p"],
            "sample": ["A"],
        }
    )
    with pytest.raises(ValueError, match="missing the 'hash_index' column"):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=["X-1"],
            pool_name="p",
        )


def test_from_samplesheet_raises_when_hash_index_not_integer_dtype():
    """from_samplesheet rejects non-integer hash_index dtypes (e.g. floats from CSV)."""
    df = pl.DataFrame(
        {
            "pool": ["p"],
            "sample": ["A"],
            "hash_index": pl.Series([1.0], dtype=pl.Float64),
        }
    )
    with pytest.raises(ValueError, match="must use an integer type"):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=["X-1"],
            pool_name="p",
        )


def test_from_samplesheet_accepts_int32_hash_index():
    """Any Polars integer dtype (e.g. Int32 from Arrow/Parquet) is valid for hash_index."""
    df = pl.DataFrame(
        {
            "pool": ["p"],
            "sample": ["A"],
            "hash_index": pl.Series([1], dtype=pl.Int32),
        }
    )
    mapping = HashedAntibodyMapping.from_samplesheet(
        df,
        all_hashing_antibodies=["X-1"],
        pool_name="p",
    )
    assert mapping["A"] == ["X-1"]


def test_from_samplesheet_raises_when_pool_column_missing():
    """from_samplesheet requires a column named exactly 'pool'."""
    df = pl.DataFrame(
        {
            "sample": ["A"],
            "hash_index": [1],
        }
    )
    with pytest.raises(ValueError, match="missing the 'pool' column"):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=["X-1"],
            pool_name="p",
        )


def test_from_samplesheet_raises_when_sample_column_missing():
    """from_samplesheet requires a column named exactly 'sample'."""
    df = pl.DataFrame(
        {
            "pool": ["p"],
            "hash_index": [1],
        }
    )
    with pytest.raises(ValueError, match="missing the 'sample' column"):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=["X-1"],
            pool_name="p",
        )


def test_from_samplesheet_raises_when_pool_has_no_matching_rows():
    """from_samplesheet fails clearly when the requested pool is absent from the sheet."""
    df = pl.DataFrame(
        {
            "pool": ["other_pool"],
            "sample": ["A"],
            "hash_index": [1],
        }
    )
    with pytest.raises(
        ValueError, match="No matching entries found in samplesheet for pool"
    ):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=["X-1"],
            pool_name="wanted_pool",
        )


def test_from_samplesheet_raises_when_duplicate_hash_index_maps_same_antibodies_to_two_samples():
    """Two samples with the same hash_index receive the same antibodies; __init__ rejects that."""
    df = pl.DataFrame(
        {
            "pool": ["p", "p"],
            "sample": ["A", "B"],
            "hash_index": [1, 1],
        }
    )
    all_hashing = {"B2M-1", "CD29-1"}
    with pytest.raises(ValueError, match="assigned to multiple samples"):
        HashedAntibodyMapping.from_samplesheet(
            df,
            all_hashing_antibodies=all_hashing,
            pool_name="p",
        )


def test_unmapped_hashing_antibodies_returns_antibodies_not_mapped_to_any_sample():
    """unmapped_hashing_antibodies returns the subset of hashing antibodies not assigned to any sample."""
    # Panel has 6 hashing antibodies (names ending with -1..-6);
    # samplesheet only lists 2 samples (hash 1 and 2)
    samplesheet = pl.DataFrame(
        {
            "pool": ["p", "p"],
            "sample": ["A", "B"],
            "hash_index": [1, 2],
        }
    )
    all_hashing = {"CD29-1", "CD29-2", "CD29-3", "B2M-1", "B2M-2", "B2M-3"}
    mapping = HashedAntibodyMapping.from_samplesheet(
        samplesheet,
        all_hashing_antibodies=all_hashing,
        pool_name="p",
    )
    # A gets -1, B gets -2 -> CD29-3 and B2M-3 are unmapped
    assert set(mapping["A"]) == {"B2M-1", "CD29-1"}
    assert set(mapping["B"]) == {"B2M-2", "CD29-2"}
    assert mapping.unmapped_hashing_antibodies == {"B2M-3", "CD29-3"}
    assert mapping.unmapped_hashing_antibodies.isdisjoint(
        {"CD29-1", "CD29-2", "B2M-1", "B2M-2"}
    )


def test_collect_hash_info(sample_hashed_pixel_files):
    """Test the sample calling functionality."""
    pxl = read(sample_hashed_pixel_files)
    cc = collect_hash_info(
        pxl,
        hashed_antibody_mapping=HashedAntibodyMapping(
            mapping={
                "PBMC": ["CD29-1", "B2M-1"],
                "Raji": ["CD29-2", "B2M-2"],
                "Jurkat": ["CD29-3", "B2M-3"],
            },
            all_hashing_antibodies=[
                "CD29-1",
                "CD29-2",
                "CD29-3",
                "B2M-1",
                "B2M-2",
                "B2M-3",
            ],
        ),
        confidence_threshold=0.8,
    )

    comp_counts = cc.group_by("called_sample").len("count")
    assert comp_counts.shape[0] == 3
    assert all(comp_counts["count"] == 10)


def test_collect_hash_info_should_use_all_hashing_antibodies(sample_hashed_pixel_files):
    """collect_hash_info uses all hashing antibodies; unmapped contribute to undetermined_hash_count."""
    pxl = read(sample_hashed_pixel_files)
    cc = collect_hash_info(
        pxl,
        hashed_antibody_mapping=HashedAntibodyMapping(
            mapping={
                "PBMC": ["CD29-1", "B2M-1"],
                "Raji": ["CD29-2", "B2M-2"],
            },
            all_hashing_antibodies=[
                "CD29-1",
                "CD29-2",
                "CD29-3",
                "B2M-1",
                "B2M-2",
                "B2M-3",
            ],
        ),
        confidence_threshold=0.8,
        undetermined_sample_name="test",
    )
    # called_sample can be PBMC, Raji, or "test" (when undetermined_hash_count wins),
    # so we get 3 groups
    comp_counts = cc.group_by("called_sample").len("count")

    assert comp_counts.shape[0] == 3
    assert "test_hash_count" in cc.columns
    assert "undetermined_hash_count" not in cc.columns
    assert comp_counts["count"].sum() == 30  # all components assigned to one of the two
    assert set(comp_counts["called_sample"].to_list()) == {
        "PBMC",
        "Raji",
        "test",
    }
    assert comp_counts.filter(pl.col("called_sample") == "test")["count"].item() == 10


def test_collect_hash_info_all_undetermined(sample_hashed_pixel_files):
    """Test is undetermined when confidence is below threshold"""
    pxl = read(sample_hashed_pixel_files)
    cc = collect_hash_info(
        pxl,
        hashed_antibody_mapping=HashedAntibodyMapping(
            mapping={
                "PBMC": ["CD29-1", "B2M-1"],
                "Raji": ["CD29-2", "B2M-2"],
                "Jurkat": ["CD29-3", "B2M-3"],
            },
            all_hashing_antibodies=[
                "CD29-1",
                "CD29-2",
                "CD29-3",
                "B2M-1",
                "B2M-2",
                "B2M-3",
            ],
        ),
        confidence_threshold=1.0,
        undetermined_sample_name="test",
    )

    assert all(cc["called_sample"] == "test")


@pytest.mark.slow
def test_sample_calling(sample_hashed_pixel_files, tmp_path):
    """Test the sample calling functionality."""
    pxl = read(sample_hashed_pixel_files)
    samplesheet, all_hashing = _samplesheet_and_hashing_for_three_samples()
    hashed_antibodies = HashedAntibodyMapping.from_samplesheet(
        samplesheet,
        all_hashing_antibodies=all_hashing,
        pool_name="test_pool",
    )
    output_folder = tmp_path
    sample_calling(
        input_pxl=pxl,
        hashing_antibody_mapping=hashed_antibodies,
        output_folder=output_folder,
        remove_incompatible=False,
    )

    output_files = list(output_folder.glob("*.dehashed.pxl"))
    assert len(output_files) == 3

    original_counts = pxl.adata().to_df()
    for file in output_files:
        dehashed_pxl = read(file)
        dehashed_adata = dehashed_pxl.adata()
        dehashed_counts = dehashed_adata.to_df()
        assert dehashed_counts.shape[0] == 10
        sample_name = list(dehashed_pxl.metadata().keys())[0]
        for ab in hashed_antibodies[sample_name]:
            base_name = ab.split("-")[0]
            assert all(
                original_counts.loc[
                    dehashed_counts.index,
                    original_counts.columns.str.contains(base_name),
                ].sum(axis=1)
                == dehashed_counts[base_name]
            )
            # We move the hash counts from the original counts into obs under
            # the name "original_hash_counts_{ab}"
            assert all(
                dehashed_adata.obs[f"original_hash_counts_{ab}"]
                == original_counts.loc[dehashed_counts.index, ab]
            )
            # The hashing antibodies should be removed from the adata
            assert (
                set(dehashed_adata.var.index).intersection(
                    hashed_antibodies.hashing_antibodies
                )
                == set()
            )


@pytest.mark.slow
def test_sample_calling_does_not_strip_suffix_from_non_hash_markers(
    tmp_path: Path,
):
    """Regression test for markers like `PD-1` being mangled to `PD`.

    The dehashing step should only strip `-<hash_index>` for *known hashing*
    antibodies, not for arbitrary biological marker IDs that happen to end
    with `-<digits>`.
    """
    panel_df = pd.DataFrame(
        [
            {
                "marker_id": "PD-1",
                "control": False,
                "uniprot_id": "P00001",
                "sequence_1": "ATCGATCGAA",
                "conj_id": "conj_pd1",
                "sequence_2": "ATCGATCGAC",
            },
            {
                "marker_id": "HashA",
                "control": False,
                "uniprot_id": "P00002",
                "sequence_1": "ATCGATCGAT",
                "conj_id": "conj_hashA",
                "sequence_2": "ATCGATCGAG",
            },
            {
                "marker_id": "HashA-1",
                "control": False,
                "uniprot_id": "P00003",
                "sequence_1": "ATCGATCGTT",
                "conj_id": "conj_hashA1",
                "sequence_2": "ATCGATCGTG",
            },
            {
                "marker_id": "HashB-2",
                "control": False,
                "uniprot_id": "P00004",
                "sequence_1": "ATCGATCGTA",
                "conj_id": "conj_hashB2",
                "sequence_2": "ATCGATCGTC",
            },
        ]
    ).set_index("marker_id")

    panel = PNAAntibodyPanel(
        df=panel_df,
        metadata=AntibodyPanelMetadata(
            name="test-panel",
            version="0.1.0",
            aliases=["test-panel"],
            description="Synthetic panel for sample-calling dehashing regression test.",
        ),
    )

    # Keep the edgelist graph connected so "stranded node" removal
    # doesn't randomly drop unrelated marker counts in this unit test.
    edgelist = pl.DataFrame(
        {
            "umi1": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "umi2": pl.Series([2, 3, 4], dtype=pl.UInt64),
            "read_count": pl.Series([10, 10, 10], dtype=pl.UInt32),
            "uei_count": pl.Series([5, 5, 5], dtype=pl.UInt32),
            "marker_1": ["PD-1", "PD-1", "HashA-1"],
            "marker_2": ["PD-1", "HashA-1", "HashA-1"],
            "component": ["c1", "c1", "c1"],
        }
    )

    target = tmp_path / "input.pxl"
    with PixelFileWriter(target) as writer:
        writer.write_edgelist(edgelist)
        con = writer.get_connection()
        adata = pna_edgelist_to_anndata(con, panel=panel)
        writer.write_adata(adata)
        writer.write_metadata(
            {
                "sample_name": "input",
                "version": "0.1.0",
                "panel_name": "custom_panel",
            }
        )

    input_pxl = read(target)
    original_counts = input_pxl.adata().to_df()

    hashing_antibodies = HashedAntibodyMapping(
        mapping={"S1": ["HashA-1"]},
        all_hashing_antibodies=["HashA-1", "HashB-2"],
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sample_calling(
        input_pxl=input_pxl,
        hashing_antibody_mapping=hashing_antibodies,
        output_folder=out_dir,
        remove_incompatible=True,
        confidence_threshold=0.5,
    )

    output_files = list(out_dir.glob("*.dehashed.pxl"))
    assert len(output_files) == 1

    dehashed_pxl = read(output_files[0])
    dehashed_counts = dehashed_pxl.adata().to_df()

    assert dehashed_counts.loc["c1", "PD-1"] == original_counts.loc["c1", "PD-1"]


@pytest.mark.slow
def test_sample_calling_with_undetermined(sample_hashed_pixel_files, tmp_path):
    """Test the sample calling functionality."""
    confidence_threshold = 0.96
    pxl = read(sample_hashed_pixel_files)
    samplesheet, all_hashing = _samplesheet_and_hashing_for_three_samples()
    hashed_antibodies = HashedAntibodyMapping.from_samplesheet(
        samplesheet,
        all_hashing_antibodies=all_hashing,
        pool_name="test_pool",
    )
    output_folder = tmp_path
    sample_calling(
        input_pxl=pxl,
        hashing_antibody_mapping=hashed_antibodies,
        output_folder=output_folder,
        remove_incompatible=True,
        confidence_threshold=confidence_threshold,
    )

    output_files = list(output_folder.glob("*.dehashed.pxl"))
    assert len(output_files) == 4

    for file in output_files:
        dehashed_pxl = read(file)
        dehashed_counts = dehashed_pxl.adata().to_df()
        sample_name = list(dehashed_pxl.metadata().keys())[0]
        if sample_name == "undetermined":
            assert dehashed_counts.shape[0] == 2
        else:
            assert all(
                dehashed_pxl.adata().obs["sample_confidence"] >= confidence_threshold
            )


def test_collect_nodes_to_remove_panel_based_incompatible():
    """_collect_nodes_to_remove uses panel hashing minus current sample (not other samples).

    Regression: works with one sample in pool; incompatible = all panel hashes
    minus current sample's, so hashes not in the samplesheet are still removed.
    """
    edgelist = pl.LazyFrame(
        {
            "component": ["c1", "c1"],
            "umi1": [1, 2],
            "umi2": [2, 3],
            "marker_1": ["h1", "h2"],
            "marker_2": ["h2", "h3"],
        }
    ).cast({"umi1": pl.UInt64, "umi2": pl.UInt64})
    all_hashing_in_panel = {"h1", "h2", "h3"}
    sample_antibodies = ["h2"]  # current sample B
    result = _collect_nodes_to_remove(
        edgelist,
        all_hashing_in_panel=all_hashing_in_panel,
        sample_antibodies=sample_antibodies,
    )
    assert "umi" in result.columns
    assert "cause" in result.columns
    # Incompatible with B are h1 and h3; edges touch these -> some nodes to remove
    assert result.height >= 1


def test_collect_nodes_to_remove_one_sample_in_pool():
    """With only one sample, incompatible = all panel hashes minus that sample's (no crash)."""
    edgelist = pl.LazyFrame(
        {
            "component": ["c1"],
            "umi1": [1],
            "umi2": [2],
            "marker_1": ["h1"],
            "marker_2": ["h1"],
        }
    ).cast({"umi1": pl.UInt64, "umi2": pl.UInt64})
    all_hashing_in_panel = {"h1", "h2", "h3"}
    sample_antibodies = ["h1"]
    result = _collect_nodes_to_remove(
        edgelist,
        all_hashing_in_panel=all_hashing_in_panel,
        sample_antibodies=sample_antibodies,
    )
    assert "umi" in result.columns
    assert "cause" in result.columns
    # Incompatible = {h2, h3}; no edges use those; single edge (1,2) stays, no stranded. No crash.
    assert result.shape[1] == 2


class _FakeFilteredDataset:
    """Minimal stand-in for ``PNAPixelDataset`` after ``filter()`` for ``create_final_report`` tests."""

    def __init__(
        self,
        *,
        component_ids: set[str],
        sample_confidences: list[float] | None = None,
    ):
        self._component_ids = component_ids
        self._sample_confidences = sample_confidences

    def components(self) -> set[str]:
        return self._component_ids

    def adata(
        self,
        add_log1p_transform: bool = True,
        add_clr_transform: bool = True,
    ) -> anndata.AnnData:
        assert self._sample_confidences is not None
        n = len(self._sample_confidences)
        # Explicit string obs index avoids AnnData ImplicitModificationWarning on index coercion.
        obs_index = [f"comp_{i}" for i in range(n)]
        return anndata.AnnData(
            X=np.zeros((n, 1)),
            obs=pd.DataFrame(
                {"sample_confidence": self._sample_confidences},
                index=obs_index,
            ),
        )


class _FakeMergedDataset:
    """Mimics a merged post-sample-calling ``PNAPixelDataset`` for unit-testing ``create_final_report``."""

    def __init__(
        self,
        *,
        all_components: set[str],
        undetermined_components: set[str] | None,
        confidences_per_sample: dict[str, list[float]],
    ):
        self._all_components = all_components
        self._undetermined_components = undetermined_components
        self._confidences_per_sample = confidences_per_sample

    def components(self) -> set[str]:
        return self._all_components

    def sample_names(self) -> set[str]:
        return set(self._confidences_per_sample.keys())

    def filter(
        self,
        samples=None,
        components=None,
        markers=None,
    ) -> _FakeFilteredDataset:
        if samples == "undetermined":
            if self._undetermined_components is None:
                raise ValueError(
                    "One or more of the specified samples do not exist in the dataset."
                )
            return _FakeFilteredDataset(
                component_ids=self._undetermined_components,
                sample_confidences=self._confidences_per_sample.get("undetermined"),
            )
        if samples not in self._confidences_per_sample:
            raise ValueError(
                "One or more of the specified samples do not exist in the dataset."
            )
        return _FakeFilteredDataset(
            component_ids=set(),
            sample_confidences=self._confidences_per_sample[samples],
        )


def test_create_final_report_works_when_no_undetermined_sample():
    """When ``undetermined`` is not a sample, the success rate is 100%."""
    ds = _FakeMergedDataset(
        all_components={"c1", "c2", "c3"},
        undetermined_components=None,
        confidences_per_sample={
            "PBMC": [0.99, 0.98],
            "Raji": [0.97],
        },
    )
    report = create_final_report(ds)  # type: ignore[arg-type]

    assert report.sample_id == "all"
    assert report.product_id == "single-cell-pna"
    assert report.report_type == "sample_calling_total"
    assert report.number_of_components == 3
    assert report.percentage_of_components_successfully_called == 1.0
    assert report.sample_confidences_per_sample == {
        "PBMC": [0.99, 0.98],
        "Raji": [0.97],
    }


def test_create_final_report_percentage_excludes_undetermined_components():
    """Success rate is 1 minus the fraction of components in the undetermined sample."""
    ds = _FakeMergedDataset(
        all_components={"a", "b", "c", "d"},
        undetermined_components={"d"},
        confidences_per_sample={
            "PBMC": [1.0, 1.0, 1.0],
            "undetermined": [0.2],
        },
    )
    report = create_final_report(ds)  # type: ignore[arg-type]

    assert report.number_of_components == 4
    assert report.percentage_of_components_successfully_called == pytest.approx(0.75)
    assert report.sample_confidences_per_sample["PBMC"] == [1.0, 1.0, 1.0]
    assert report.sample_confidences_per_sample["undetermined"] == [0.2]


def test_create_final_report_zero_success_when_all_components_undetermined():
    """When every component belongs to ``undetermined``, the success rate is 0."""
    ds = _FakeMergedDataset(
        all_components={"x", "y"},
        undetermined_components={"x", "y"},
        confidences_per_sample={"undetermined": [0.1, 0.2]},
    )
    report = create_final_report(ds)  # type: ignore[arg-type]

    assert report.number_of_components == 2
    assert report.percentage_of_components_successfully_called == 0.0
    assert report.sample_confidences_per_sample["undetermined"] == [0.1, 0.2]


def test_warn_if_undetermined_has_high_confidence_logs_when_fraction_above_five_percent(
    caplog,
):
    """More than 5% strictly above the threshold should emit one WARNING."""
    with caplog.at_level(
        logging.WARNING, logger="pixelator.pna.sample_calling.sample_calling"
    ):
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=np.array([0.95] * 6 + [0.1] * 94),
            confidence_threshold=0.9,
        )
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "samplesheet" in caplog.text


def test_warn_if_undetermined_has_high_confidence_no_log_when_fraction_is_exactly_five_percent(
    caplog,
):
    """The check uses ``> 0.05``, so exactly 5% above threshold must not warn."""
    with caplog.at_level(
        logging.WARNING, logger="pixelator.pna.sample_calling.sample_calling"
    ):
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=np.array([0.95] * 5 + [0.1] * 95),
            confidence_threshold=0.9,
        )
    assert caplog.records == []


def test_warn_if_undetermined_has_high_confidence_no_log_when_all_at_or_below_threshold(
    caplog,
):
    """Values equal to the threshold are not counted as high confidence (strict ``>``)."""
    with caplog.at_level(
        logging.WARNING, logger="pixelator.pna.sample_calling.sample_calling"
    ):
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=np.full(50, 0.9),
            confidence_threshold=0.9,
        )
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=np.array([0.1, 0.2, 0.5]),
            confidence_threshold=0.9,
        )
    assert caplog.records == []


def test_warn_if_undetermined_has_high_confidence_logs_for_single_high_value(caplog):
    """One component above threshold is 100% of the undetermined set, which is > 5%."""
    with caplog.at_level(
        logging.WARNING, logger="pixelator.pna.sample_calling.sample_calling"
    ):
        warn_if_undetermined_has_high_confidence(
            undetermined_sample_confidences=np.array([0.99]),
            confidence_threshold=0.9,
        )
    assert len(caplog.records) == 1
