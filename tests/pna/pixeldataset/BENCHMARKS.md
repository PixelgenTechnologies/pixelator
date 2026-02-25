# PNAPixelDataset accessor benchmarks (subsetting focus)

Benchmarks for **subsetting** (filter by components/samples) and **filtered reads** (edgelist, proximity, layouts) after the lazy DuckDB→Polars refactor (PNA-1761). All benchmarks work on **10 components per sample** so that subsetting is performant and correctly exercised.

## How to run

**Single run (current branch):**

```bash
uv run pytest --benchmark-enable --benchmark-only tests/pna/pixeldataset/test_benchmark_pixeldataset.py -v
```

Or use the project task:

```bash
uv run task test-benchmark -- tests/pna/pixeldataset/test_benchmark_pixeldataset.py
```

**Compare current branch vs dev (run on both, then diff):**

```bash
./scripts/run_pixeldataset_benchmark_compare.sh
```

This runs the benchmarks on your current branch, switches to `dev`, runs them again, then switches back. The dev run is compared against the current-branch run so you get a side-by-side table. Requires the two PNA062 `.pxl` files in the repo root.

## Data

- **Source**: Realistic `.pxl` files from the **repo root** (required to run):
  - `PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl`
  - `PNA062_PHA_PBMCs_1000cells_S04_S4.layout.pxl`
- **Subsetting**: Benchmarks use **10 components per sample**. Single-sample tests use 10 components; multi-sample use 10 from each sample (20 total). This keeps the working set small and emphasizes subsetting and filtered read performance.
- If the files are not found, benchmark tests are skipped.

## What is benchmarked

- **Subsetting + read**: `filter(components=[...])` then `.edgelist().to_polars()` (and same for proximity, layouts). Measures cost of applying the filter and performing the read.
- **Pre-filtered reads**: Dataset already restricted to 10 components; time `.edgelist().to_polars()`, `.proximity().to_polars()`, `.precomputed_layouts().to_polars()`.
- **Multi-sample**: Same patterns with two samples; one test filters by sample then reads edgelist.

## Memory

Peak memory is not measured. For memory regression testing, snapshot RSS (e.g. `psutil`) or use `tracemalloc` around the same accessors.
