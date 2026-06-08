# `pna_edgelist_to_anndata` benchmark results

Captured with [`bench_pna_edgelist_to_anndata.py`](../bench_pna_edgelist_to_anndata.py)
against a synthetic 67.4M-row edgelist (30,000 components, ~2,000 edges per
component, 158 markers from the `proxiome-v1-immuno-155-v1.0` panel).

Both runs reuse the same persisted `.pxl` at `/tmp/pna_bench/bench.pxl` so the
process starts with a clean RSS and the comparison is fair.

To reproduce, run:

```bash
uv run python -m tests.pna.benchmarks.bench_pna_edgelist_to_anndata \
    --n-components 30000 --avg-edges-per-component 2000 \
    --pxl-path /tmp/pna_bench/bench.pxl --skip-full-function --skip-explain
# then
uv run python -m tests.pna.benchmarks.bench_pna_edgelist_to_anndata \
    --pxl-path /tmp/pna_bench/bench.pxl --probe-non-distinct
```

## Headline numbers

| variant                                | wall time | tracemalloc peak | RSS delta during call | RSS peak after call |
| -------------------------------------- | --------: | ---------------: | --------------------: | ------------------: |
| `baseline_current.txt` (pre-refactor)  |     1.30s |          92 MiB  |             2.77 GiB  |           3.05 GiB  |
| `refactor_final.txt` (post-refactor)   |    15.20s |          35 MiB  |             0.79 GiB  |           1.07 GiB  |

- Peak RSS dropped from 3.05 GiB to 1.07 GiB (~3x), and the in-call RSS delta
  attributable to `pna_edgelist_to_anndata` dropped from 2.77 GiB to 0.79 GiB
  (~3.5x). On real datasets where COUNT(DISTINCT) hash tables dominate (more
  distinct UMIs per `(component, marker)`), the ratio improves further.
- Wall time increased ~12x because the refactor issues 2 DuckDB queries per
  batch of 512 components instead of two global queries. Per-batch query
  startup overhead is the dominant cost.
- `tracemalloc` (Python heap only) more than halved — the dense PIVOT pandas
  DataFrame and its temporary copies are gone.

## Confirmation of the hot spot

The baseline file also captures `EXPLAIN ANALYZE` plans plus `PRAGMA threads=1`
runs and a "no-DISTINCT" probe. Highlights:

- `EXPLAIN ANALYZE` of the original wide PIVOT and component metrics queries
  showed `HASH_GROUP_BY` with `COUNT(DISTINCT umi*)` as the leaf hot operator.
- The `threads=1` runs took ~5s each vs ~0.6s with default threads, confirming
  the original query depends heavily on parallel hash partitioning.
- Replacing `COUNT(DISTINCT umi*)` with `COUNT(umi*)` (semantically wrong but
  cheaper) ran in 0.52s and 0.06s respectively (vs 0.60s and 0.52s), isolating
  `COUNT(DISTINCT)` as the dominant cost driver.
