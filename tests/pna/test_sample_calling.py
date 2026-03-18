"""Tests for the sample calling module.

Copyright © 2025 Pixelgen Technologies AB.
"""

import numpy as np
import polars as pl
import pytest

from pixelator.pna import read
from pixelator.pna.sample_calling import collect_hash_info, sample_calling
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
    import anndata

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
    """Samplesheet and full hashing list for PBMC/Raji/Jurkat (hash 1/2/3). Use for tests that need HashedAntibodyMapping."""
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
    # Only from_samplesheet is supported; from_dict is removed so both samplesheet and all_hashing_antibodies are always required
    assert not hasattr(HashedAntibodyMapping, "from_dict"), (
        "from_dict must be removed; use from_samplesheet(samplesheet, all_hashing_antibodies, pool_name)"
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


def test_unmapped_hashing_antibodies_returns_antibodies_not_mapped_to_any_sample():
    """unmapped_hashing_antibodies returns the subset of hashing antibodies not assigned to any sample."""
    # Panel has 6 hashing antibodies (names ending with -1..-6); samplesheet only lists 2 samples (hash 1 and 2)
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
    )
    # called_sample can be PBMC, Raji, or "undetermined" (when undetermined_hash_count wins), so we get 3 groups
    comp_counts = cc.group_by("called_sample").len("count")
    assert comp_counts.shape[0] == 3
    assert "undetermined_hash_count" in cc.columns
    assert comp_counts["count"].sum() == 30  # all components assigned to one of the two
    assert set(comp_counts["called_sample"].to_list()) == {
        "PBMC",
        "Raji",
        "undetermined",
    }
    assert (
        comp_counts.filter(pl.col("called_sample") == "undetermined")["count"].item()
        == 10
    )

    # Check the sample calling confidence


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
        save_undetermined=True,
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
