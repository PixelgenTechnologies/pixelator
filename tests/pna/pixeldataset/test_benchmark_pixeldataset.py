"""Benchmarks for core PNAPixelDataset loading APIs.

Copyright © 2025 Pixelgen Technologies AB.
"""

from pixelator.pna.pixeldataset import PNAPixelDataset


def test_benchmark_load_edgelist(benchmark, pxl_file):
    dataset = PNAPixelDataset.from_pxl_files(pxl_file)
    result = benchmark(lambda: dataset.edgelist().to_polars())
    assert result.height > 0


def test_benchmark_load_proximity(benchmark, pxl_file):
    dataset = PNAPixelDataset.from_pxl_files(pxl_file)
    result = benchmark(lambda: dataset.proximity().to_polars())
    assert result.height > 0


def test_benchmark_load_precomputed_layouts(benchmark, pxl_file):
    dataset = PNAPixelDataset.from_pxl_files(pxl_file)
    result = benchmark(lambda: dataset.precomputed_layouts().to_polars())
    assert result.height > 0
