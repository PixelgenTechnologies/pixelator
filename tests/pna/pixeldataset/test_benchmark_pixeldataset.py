"""Benchmarks for PNAPixelDataset accessors with subsetting (10 components per sample).

Uses realistic-sized .pxl files from the repo root, filtered to 10 components per sample
so that subsetting and filtered read performance are the focus:
  - PNA062_PHA_PBMCs_1000cells_S04_S4.layout.pxl
  - PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl

Run with: uv run pytest --benchmark-enable --benchmark-only tests/pna/pixeldataset/test_benchmark_pixeldataset.py
See BENCHMARKS.md for more.
"""

from pathlib import Path

import pytest

from pixelator.pna.pixeldataset import PNAPixelDataset

N_COMPONENTS_PER_SAMPLE = 10

# Repo root (tests/pna/pixeldataset -> 3 levels up)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PXL_PHA = _REPO_ROOT / "PNA062_PHA_PBMCs_1000cells_S04_S4.layout.pxl"
_PXL_UNSTIM = _REPO_ROOT / "PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl"


def _require_benchmark_files():
    if not _PXL_UNSTIM.exists():
        pytest.skip(
            f"Benchmark data not found: {_PXL_UNSTIM}. "
            "Add PNA062 .layout.pxl files to repo root to run benchmarks."
        )
    if not _PXL_PHA.exists():
        pytest.skip(
            f"Benchmark data not found: {_PXL_PHA}. "
            "Add PNA062 .layout.pxl files to repo root to run benchmarks."
        )


def _first_n_components(dataset: PNAPixelDataset, n: int):
    """First n component names (any order)."""
    return set(list(dataset.components())[:n])


def _first_n_components_per_sample(dataset: PNAPixelDataset, n: int):
    """First n component names from each sample (n per sample)."""
    adata = dataset.adata(add_log1p_transform=False, add_clr_transform=False)
    components = set()
    for sample_name in dataset.sample_names():
        sample_components = adata.obs[adata.obs["sample"] == sample_name].index
        components.update(sample_components[:n].tolist())
    return components


@pytest.fixture(scope="session")
def full_dataset_single():
    """Full single-sample dataset (loaded once per session)."""
    _require_benchmark_files()
    return PNAPixelDataset.from_pxl_files(_PXL_UNSTIM)


@pytest.fixture(scope="session")
def full_dataset_multi():
    """Full two-sample dataset (loaded once per session)."""
    _require_benchmark_files()
    return PNAPixelDataset.from_pxl_files([_PXL_UNSTIM, _PXL_PHA])


@pytest.fixture(scope="session")
def pxl_dataset_10_components(full_dataset_single: PNAPixelDataset):
    """Single-sample dataset restricted to 10 components (PNA062 unstim)."""
    components = _first_n_components(full_dataset_single, N_COMPONENTS_PER_SAMPLE)
    return full_dataset_single.filter(components=components)


@pytest.fixture(scope="session")
def multi_sample_dataset_10_components(full_dataset_multi: PNAPixelDataset):
    """Two-sample dataset restricted to 10 components per sample (unstim + PHA)."""
    components = _first_n_components_per_sample(
        full_dataset_multi, N_COMPONENTS_PER_SAMPLE
    )
    return full_dataset_multi.filter(components=components)


# --- Subsetting + access (measure filter then read) ---


def test_benchmark_filter_10_components_then_edgelist(
    benchmark, full_dataset_single: PNAPixelDataset
):
    """Time: apply component filter then read edgelist (subsetting + read)."""
    components = _first_n_components(full_dataset_single, N_COMPONENTS_PER_SAMPLE)
    components_list = list(components)

    def run():
        return (
            full_dataset_single.filter(components=components_list)
            .edgelist()
            .to_polars()
        )

    result = benchmark(run)
    assert result.shape[0] > 0
    assert result["component"].n_unique() <= N_COMPONENTS_PER_SAMPLE


def test_benchmark_filter_10_components_then_proximity(
    benchmark, full_dataset_single: PNAPixelDataset
):
    """Time: apply component filter then read proximity."""
    components = _first_n_components(full_dataset_single, N_COMPONENTS_PER_SAMPLE)
    components_list = list(components)

    def run():
        return (
            full_dataset_single.filter(components=components_list)
            .proximity()
            .to_polars()
        )

    result = benchmark(run)
    assert result.shape[0] >= 0


def test_benchmark_filter_10_components_then_layouts(
    benchmark, full_dataset_single: PNAPixelDataset
):
    """Time: apply component filter then read layouts (no marker counts)."""
    components = _first_n_components(full_dataset_single, N_COMPONENTS_PER_SAMPLE)
    components_list = list(components)

    def run():
        return (
            full_dataset_single.filter(components=components_list)
            .precomputed_layouts(add_marker_counts=False)
            .to_polars()
        )

    result = benchmark(run)
    assert result.shape[0] > 0


def test_benchmark_multi_filter_10_per_sample_then_edgelist(
    benchmark, full_dataset_multi: PNAPixelDataset
):
    """Time: filter to 10 components per sample then read edgelist (multi-sample)."""
    components = _first_n_components_per_sample(
        full_dataset_multi, N_COMPONENTS_PER_SAMPLE
    )
    components_list = list(components)

    def run():
        return (
            full_dataset_multi.filter(components=components_list).edgelist().to_polars()
        )

    result = benchmark(run)
    assert result.shape[0] > 0


# --- Pre-filtered dataset: read accessors (measure read performance on 10 components) ---


def test_benchmark_edgelist_to_polars(
    benchmark, pxl_dataset_10_components: PNAPixelDataset
):
    """Time: edgelist read on already-filtered (10 components) dataset."""

    def run():
        return pxl_dataset_10_components.edgelist().to_polars()

    result = benchmark(run)
    assert result.shape[0] > 0


def test_benchmark_edgelist_to_df(
    benchmark, pxl_dataset_10_components: PNAPixelDataset
):
    """Time: edgelist to pandas on already-filtered dataset."""

    def run():
        return pxl_dataset_10_components.edgelist().to_df()

    result = benchmark(run)
    assert len(result) > 0


def test_benchmark_proximity_to_polars(
    benchmark, pxl_dataset_10_components: PNAPixelDataset
):
    """Time: proximity read on already-filtered dataset."""

    def run():
        return pxl_dataset_10_components.proximity().to_polars()

    result = benchmark(run)
    assert result.shape[0] >= 0


def test_benchmark_layouts_to_polars_no_marker_counts(
    benchmark, pxl_dataset_10_components: PNAPixelDataset
):
    """Time: layouts read (no marker counts) on already-filtered dataset."""

    def run():
        return pxl_dataset_10_components.precomputed_layouts(
            add_marker_counts=False
        ).to_polars()

    result = benchmark(run)
    assert result.shape[0] > 0


def test_benchmark_layouts_to_polars_with_marker_counts(
    benchmark, pxl_dataset_10_components: PNAPixelDataset
):
    """Time: layouts read (with marker counts) on already-filtered dataset."""

    def run():
        return pxl_dataset_10_components.precomputed_layouts(
            add_marker_counts=True
        ).to_polars()

    result = benchmark(run)
    assert result.shape[0] > 0


# --- Multi-sample pre-filtered ---


def test_benchmark_multi_edgelist_to_polars(
    benchmark, multi_sample_dataset_10_components: PNAPixelDataset
):
    """Time: edgelist read on two samples, 10 components per sample."""

    def run():
        return multi_sample_dataset_10_components.edgelist().to_polars()

    result = benchmark(run)
    assert result.shape[0] > 0


def test_benchmark_multi_filter_by_sample_then_edgelist(
    benchmark, multi_sample_dataset_10_components: PNAPixelDataset
):
    """Time: filter by sample then read edgelist (subsetting by sample + read)."""
    one_sample = next(iter(multi_sample_dataset_10_components.sample_names()))

    def run():
        return (
            multi_sample_dataset_10_components.filter(samples=[one_sample])
            .edgelist()
            .to_polars()
        )

    result = benchmark(run)
    assert result.shape[0] > 0
