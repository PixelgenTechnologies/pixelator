"""Standalone benchmark for ``pna_edgelist_to_anndata``.

Copyright © 2026 Pixelgen Technologies AB.

This is an opt-in, manually run script. It is intentionally *not* picked up by
pytest collection (the surrounding module has no ``test_*`` functions) so it
will only execute when invoked directly, e.g.::

    uv run python -m tests.pna.benchmarks.bench_pna_edgelist_to_anndata

The goal is to confirm that ``COUNT(DISTINCT umi1)`` / ``COUNT(DISTINCT umi2)``
in :func:`pixelator.pna.anndata.pna_edgelist_to_anndata` is the dominant memory
consumer when the function is run against a large edgelist, and to provide a
baseline that can be re-run after the per-component refactor to confirm a
memory reduction.

Measurements collected per run
------------------------------
* Wall-clock time
* ``tracemalloc`` peak (Python heap allocations only)
* ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (process peak RSS, including
  DuckDB's hash tables which live outside the Python heap)

Variants exercised
------------------
1. The full function as it ships.
2. ``EXPLAIN ANALYZE`` for each of the two SQL strings currently used by the
   function (the wide ``PIVOT`` query and the per-component metrics query).
3. The same two queries with ``PRAGMA threads = 1`` to demonstrate that the
   per-thread distinct hash tables of ``COUNT(DISTINCT umi*)`` amplify peak
   memory roughly linearly with thread count.
4. (Optional sanity probe) The two queries with ``COUNT(DISTINCT umi*)``
   replaced by ``COUNT(umi*)``; semantically wrong, but it isolates the
   distinct aggregation as the cost driver if peak RSS drops sharply.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import resource
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb
import numpy as np
import polars as pl

from pixelator import __version__
from pixelator.pna.anndata import pna_edgelist_to_anndata
from pixelator.pna.config import load_antibody_panel, pna_config
from pixelator.pna.pixeldataset.io import PixelFileWriter

logger = logging.getLogger("bench_pna_edgelist_to_anndata")


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _ru_maxrss_bytes() -> int:
    """Return ``ru_maxrss`` in bytes, accounting for platform differences.

    Linux reports kilobytes, macOS reports bytes. We normalise to bytes.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)
    return int(raw) * 1024


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = min(int(math.log(n, 1024)), len(units) - 1)
    return f"{n / 1024**i:.2f} {units[i]}"


@dataclass
class Measurement:
    label: str
    wall_seconds: float
    tracemalloc_peak: int
    rss_delta: int
    rss_peak_after: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"{self.label:<55s} "
            f"time={self.wall_seconds:7.2f}s  "
            f"tracemalloc_peak={_fmt_bytes(self.tracemalloc_peak):>10s}  "
            f"rss_delta={_fmt_bytes(self.rss_delta):>10s}  "
            f"rss_peak={_fmt_bytes(self.rss_peak_after):>10s}"
        )


def _measure(label: str, func: Callable[[], object]) -> tuple[Measurement, object]:
    """Run ``func`` while recording wall time, tracemalloc peak and RSS delta."""
    gc.collect()
    rss_before = _ru_maxrss_bytes()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    rss_after = _ru_maxrss_bytes()
    return (
        Measurement(
            label=label,
            wall_seconds=elapsed,
            tracemalloc_peak=peak,
            rss_delta=max(0, rss_after - rss_before),
            rss_peak_after=rss_after,
        ),
        result,
    )


# ---------------------------------------------------------------------------
# Synthetic edgelist generator
# ---------------------------------------------------------------------------


def _generate_edgelist(
    n_components: int,
    avg_edges_per_component: int,
    markers: list[str],
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic edgelist with the same schema as a real PNA edgelist.

    The graph for each component is bipartite: A-side nodes have a fixed
    ``marker_1`` and B-side nodes have a fixed ``marker_2``. UMIs are
    monotonically increasing integers across the whole edgelist so they are
    globally unique. Each node typically participates in several edges, which
    exercises the ``COUNT(DISTINCT umi*)`` operator without making it trivially
    cheap (i.e. distinct UMIs are still a large fraction of total rows).
    """
    rng = np.random.default_rng(seed)

    # Per-component sizes roughly cluster around the requested average so that
    # the largest component is meaningfully larger than the median.
    edges_per_comp = rng.integers(
        low=max(1, avg_edges_per_component // 4),
        high=max(2, avg_edges_per_component * 2),
        size=n_components,
    )

    n_markers = len(markers)
    total_edges = int(edges_per_comp.sum())

    component_col = np.empty(total_edges, dtype=np.int64)
    umi1_col = np.empty(total_edges, dtype=np.int64)
    umi2_col = np.empty(total_edges, dtype=np.int64)
    marker_1_col = np.empty(total_edges, dtype=np.int32)
    marker_2_col = np.empty(total_edges, dtype=np.int32)
    read_count_col = rng.integers(low=1, high=10, size=total_edges, dtype=np.int32)

    global_umi1_next = 0
    global_umi2_next = 0
    row = 0
    for c in range(n_components):
        n_edges = int(edges_per_comp[c])
        # A- and B-side node counts: enough that UMIs are distinct-heavy but
        # each node still appears on several edges on average.
        n_a = max(1, n_edges // 3)
        n_b = max(1, n_edges // 3)

        a_umi_ids = np.arange(global_umi1_next, global_umi1_next + n_a, dtype=np.int64)
        b_umi_ids = np.arange(global_umi2_next, global_umi2_next + n_b, dtype=np.int64)
        global_umi1_next += n_a
        global_umi2_next += n_b

        a_markers = rng.integers(low=0, high=n_markers, size=n_a, dtype=np.int32)
        b_markers = rng.integers(low=0, high=n_markers, size=n_b, dtype=np.int32)

        edge_a = rng.integers(low=0, high=n_a, size=n_edges)
        edge_b = rng.integers(low=0, high=n_b, size=n_edges)

        end = row + n_edges
        component_col[row:end] = c
        umi1_col[row:end] = a_umi_ids[edge_a]
        umi2_col[row:end] = b_umi_ids[edge_b]
        marker_1_col[row:end] = a_markers[edge_a]
        marker_2_col[row:end] = b_markers[edge_b]
        row = end

    marker_array = np.asarray(markers, dtype=object)
    return pl.DataFrame(
        {
            "component": pl.Series(component_col).cast(pl.Utf8),
            "umi1": umi1_col,
            "umi2": umi2_col,
            "marker_1": pl.Series(marker_array[marker_1_col]),
            "marker_2": pl.Series(marker_array[marker_2_col]),
            "read_count": read_count_col,
        }
    )


def _write_pxl(edgelist: pl.DataFrame, path: Path) -> None:
    with PixelFileWriter(path) as writer:
        writer.write_metadata(
            {
                "sample_name": "bench",
                "version": __version__,
                "technology": "single-cell-pna",
                "panel_name": "proxiome-v1-immuno-155-v1.0",
                "panel_version": "mock.version",
            }
        )
        writer.write_edgelist(edgelist)


# ---------------------------------------------------------------------------
# SQL we want to profile
# ---------------------------------------------------------------------------


def _wide_pivot_sql(markers: list[str], count_aggregate: str = "COUNT(DISTINCT") -> str:
    """Return the wide PIVOT query exactly as the production function builds it.

    ``count_aggregate`` is exposed so we can swap ``COUNT(DISTINCT`` for plain
    ``COUNT(`` as the optional sanity probe.
    """
    marker_names_sql = ", ".join(f"'{m}'" for m in markers)
    return f"""
        SELECT *
        FROM (
            WITH counts_df_long AS (
                WITH
                    marker_1_counts AS (
                        SELECT component, marker_1 AS marker, {count_aggregate} umi1) AS marker_1_count
                        FROM edgelist
                        GROUP BY component, marker_1),
                    marker_2_counts AS (
                        SELECT component, marker_2 AS marker, {count_aggregate} umi2) AS marker_2_count
                        FROM edgelist
                        GROUP BY component, marker_2
                    )
                SELECT
                    COALESCE(a.component, b.component) AS component,
                    COALESCE(a.marker, b.marker) AS marker,
                    COALESCE(a.marker_1_count, 0) AS marker_1_count,
                    COALESCE(b.marker_2_count, 0) AS marker_2_count,
                    COALESCE(a.marker_1_count, 0) + COALESCE(b.marker_2_count, 0) AS count
                FROM marker_1_counts a
                FULL OUTER JOIN marker_2_counts b
                    ON a.component = b.component AND a.marker = b.marker
            )
            PIVOT counts_df_long
            ON marker IN ({marker_names_sql})
            USING SUM(count)
            GROUP BY component
        )
    """


def _component_metrics_sql(count_aggregate: str = "COUNT(DISTINCT") -> str:
    return f"""
        WITH
            marker_1_counts AS (
                SELECT component, marker_1 AS marker, {count_aggregate} umi1) AS marker_1_count
                FROM edgelist
                GROUP BY component, marker_1),
            marker_2_counts AS (
                SELECT component, marker_2 AS marker, {count_aggregate} umi2) AS marker_2_count
                FROM edgelist
                GROUP BY component, marker_2
            ),
            component_marker_counts AS (
                SELECT
                    COALESCE(a.component, b.component) AS component,
                    COALESCE(a.marker_1_count, 0) AS marker_1_count,
                    COALESCE(b.marker_2_count, 0) AS marker_2_count
                FROM marker_1_counts a
                FULL OUTER JOIN marker_2_counts b
                    ON a.component = b.component AND a.marker = b.marker
            ),
            component_umi AS (
                SELECT
                    component,
                    SUM(marker_1_count) AS n_umi1,
                    SUM(marker_2_count) AS n_umi2
                FROM component_marker_counts
                GROUP BY component
            ),
            edge_counts AS (
                SELECT component, COUNT(*) AS n_edges, SUM(read_count) AS reads_in_component
                FROM edgelist
                GROUP BY component
            )
        SELECT
            u.component,
            n_umi1,
            n_umi2,
            e.n_edges,
            e.reads_in_component,
            (n_umi1 + n_umi2) AS n_umi
        FROM component_umi u
        LEFT JOIN edge_counts e ON u.component = e.component
        ORDER BY u.component
    """


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------


def _bench_full_function(pxl_path: Path) -> None:
    panel = load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")

    with PixelFileWriter(pxl_path) as writer:
        con = writer.get_connection()
        measurement, adata = _measure(
            "pna_edgelist_to_anndata (current impl)",
            lambda: pna_edgelist_to_anndata(con, panel),
        )
    print(measurement)
    print(
        f"  -> adata shape={adata.shape} "
        f"X dtype={adata.X.dtype} "
        f"n_obs_cols={len(adata.obs.columns)}"
    )


def _bench_explain(pxl_path: Path, markers: list[str], threads: int | None) -> None:
    """Run ``EXPLAIN ANALYZE`` for the two queries, optionally pinning threads."""
    con = duckdb.connect(str(pxl_path))
    try:
        if threads is not None:
            con.execute(f"PRAGMA threads = {threads}")
        thread_label = "default" if threads is None else str(threads)

        for query_label, sql in [
            ("wide PIVOT (default COUNT DISTINCT)", _wide_pivot_sql(markers)),
            (
                "component metrics (default COUNT DISTINCT)",
                _component_metrics_sql(),
            ),
        ]:
            label = f"EXPLAIN ANALYZE :: {query_label} :: threads={thread_label}"
            measurement, rows = _measure(
                label, lambda s=sql: con.execute("EXPLAIN ANALYZE " + s).fetchall()
            )
            print(measurement)
            print("---- query plan (truncated to first 60 lines) ----")
            joined = "\n".join(line for row in rows for line in row)
            for line in joined.splitlines()[:60]:
                print(line)
            print("---- end of plan ----\n")
    finally:
        con.close()


def _bench_explain_probe_non_distinct(pxl_path: Path, markers: list[str]) -> None:
    """Optional sanity probe: replace COUNT(DISTINCT ...) with COUNT(...).

    Semantics are wrong, but if peak RSS / time drops sharply this confirms the
    ``COUNT(DISTINCT)`` operator is the dominant cost.
    """
    con = duckdb.connect(str(pxl_path))
    try:
        for query_label, sql in [
            (
                "wide PIVOT (COUNT() - no distinct, WRONG semantics)",
                _wide_pivot_sql(markers, count_aggregate="COUNT("),
            ),
            (
                "component metrics (COUNT() - no distinct, WRONG semantics)",
                _component_metrics_sql(count_aggregate="COUNT("),
            ),
        ]:
            label = f"PROBE :: {query_label}"
            measurement, _ = _measure(label, lambda s=sql: con.execute(s).fetchall())
            print(measurement)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-components",
        type=int,
        default=10_000,
        help="Number of components to synthesize (default: 10000).",
    )
    parser.add_argument(
        "--avg-edges-per-component",
        type=int,
        default=1_000,
        help="Average edges per component (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--pxl-path",
        type=Path,
        default=None,
        help=(
            "Optional path to write the synthetic .pxl. If omitted a temp "
            "directory is created and removed at the end of the run."
        ),
    )
    parser.add_argument(
        "--skip-full-function",
        action="store_true",
        help="Skip running the full pna_edgelist_to_anndata function.",
    )
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip EXPLAIN ANALYZE of the two SQL queries.",
    )
    parser.add_argument(
        "--skip-threads1",
        action="store_true",
        help="Skip the PRAGMA threads=1 explain runs.",
    )
    parser.add_argument(
        "--probe-non-distinct",
        action="store_true",
        help="Run the optional COUNT() (no DISTINCT) sanity probe.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    panel = load_antibody_panel(pna_config, "proxiome-v1-immuno-155-v1.0")
    markers = panel.markers

    cleanup_tmp: tempfile.TemporaryDirectory | None = None
    if args.pxl_path is None:
        cleanup_tmp = tempfile.TemporaryDirectory(prefix="bench_pna_anndata_")
        pxl_path = Path(cleanup_tmp.name) / "bench.pxl"
    else:
        pxl_path = args.pxl_path

    try:
        if pxl_path.exists() and args.pxl_path is not None:
            logger.info(
                "Reusing existing pxl at %s (size on disk: %s) "
                "(skipping synthetic edgelist generation)",
                pxl_path,
                _fmt_bytes(os.path.getsize(pxl_path)),
            )
        else:
            logger.info(
                "Generating synthetic edgelist: n_components=%d "
                "avg_edges_per_component=%d markers=%d",
                args.n_components,
                args.avg_edges_per_component,
                len(markers),
            )
            edgelist = _generate_edgelist(
                n_components=args.n_components,
                avg_edges_per_component=args.avg_edges_per_component,
                markers=markers,
                seed=args.seed,
            )
            logger.info(
                "Generated edgelist: rows=%d size_in_memory~%s",
                edgelist.height,
                _fmt_bytes(edgelist.estimated_size()),
            )
            logger.info("Writing pxl to %s", pxl_path)
            _write_pxl(edgelist, pxl_path)
            del edgelist
            gc.collect()
            logger.info("pxl size on disk: %s", _fmt_bytes(os.path.getsize(pxl_path)))

        print()
        print("=" * 80)
        print(
            f"Benchmark for pna_edgelist_to_anndata "
            f"(n_components={args.n_components}, "
            f"avg_edges_per_component={args.avg_edges_per_component}, "
            f"markers={len(markers)})"
        )
        print("=" * 80)

        if not args.skip_full_function:
            _bench_full_function(pxl_path)
            print()

        if not args.skip_explain:
            _bench_explain(pxl_path, markers, threads=None)
            if not args.skip_threads1:
                _bench_explain(pxl_path, markers, threads=1)

        if args.probe_non_distinct:
            _bench_explain_probe_non_distinct(pxl_path, markers)

        print("Done.")
        return 0
    finally:
        if cleanup_tmp is not None:
            cleanup_tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
