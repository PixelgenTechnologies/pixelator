"""Copyright © 2026 Pixelgen Technologies AB."""

import os
from pathlib import Path

import polars as pl
import pytest
from joblib.externals.loky import get_reusable_executor

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.pna.analysis_engine import AnalysisManager, PerComponentTask
from pixelator.pna.pixeldataset import PixelDatasetSaver, PNAPixelDataset


class ThreadProbeTask(PerComponentTask):
    """A task that reports the thread configuration of the worker it runs in."""

    TASK_NAME = "thread_probe_task"

    def run_from_component_id(self, component_id: str):
        """Run from component id.

        Args:
            component_id: Component id.
        """
        pl.DataFrame({"a": [1, 2, 3]}).group_by("a").len()
        with connect_duckdb() as con:
            duckdb_threads = con.execute(
                "SELECT current_setting('threads')"
            ).fetchone()[0]  # type: ignore
        return {
            "pid": os.getpid(),
            "polars_pool": pl.thread_pool_size(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "duckdb_threads": duckdb_threads,
        }


@pytest.mark.parametrize("previous_polars_max_threads", [None, "7"])
def test_parallel_workers_use_one_thread(monkeypatch, previous_polars_max_threads):
    """Verify each loky worker gets one Polars thread and one BLAS thread."""
    get_reusable_executor().shutdown(wait=True)
    if previous_polars_max_threads is None:
        monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)
    else:
        monkeypatch.setenv("POLARS_MAX_THREADS", previous_polars_max_threads)
    parent_pool_size = pl.thread_pool_size()

    manager = AnalysisManager([ThreadProbeTask()], n_cores=2)  # type: ignore
    results = manager._execute_computations_in_parallel(
        [f"component{i}" for i in range(8)]
    )
    probes = [probe for _, probe in results]

    assert len(probes) == 8
    assert all(probe["pid"] != os.getpid() for probe in probes)
    assert len({probe["pid"] for probe in probes}) <= 2
    assert all(probe["polars_pool"] == 1 for probe in probes)
    assert all(probe["omp_num_threads"] == "1" for probe in probes)
    assert all(probe["duckdb_threads"] == 1 for probe in probes)
    assert "PIXELATOR_DUCKDB_THREADS" not in os.environ
    assert pl.thread_pool_size() == parent_pool_size
    assert os.environ.get("POLARS_MAX_THREADS") == previous_polars_max_threads


class FailingTask(PerComponentTask):
    """A task that simulates failure during execution."""

    TASK_NAME = "failing_task"

    def __init__(self, work_folder: Path | None = None):
        """Initialize the instance.

        Args:
            work_folder: Work folder.
        """
        super().__init__()
        self._setup_was_called = False
        self._teardown_was_called = False

    def setup(self) -> None:
        """Provide setup for tests.

        Returns:
            Result (None).
        """
        self._setup_was_called = True

    def teardown(self) -> None:
        """Provide teardown for tests.

        Returns:
            Result (None).
        """
        self._teardown_was_called = True

    def run_from_component_id(self, component_id: str):
        """Run from component id.

        Args:
            component_id: Component id.
        """
        raise RuntimeError("Simulated failure")


def test_tempfile_cleanup_on_failure(pna_pxl_dataset: PNAPixelDataset, tmp_path):
    """Verify tempfile cleanup on failure.

    Args:
        tmp_path: tmp path.
        pna_pxl_dataset: Pna pxl dataset.
    """
    mock_task = FailingTask()  # type: ignore
    manager = AnalysisManager([mock_task])
    pna_pixel_filtered = pna_pxl_dataset.filter(
        components=pna_pxl_dataset.adata().obs.index[:2]
    )

    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pixel_filtered).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    try:
        dataset = manager.execute_from_path(pxl_file_target.path, pxl_file_target)
    except RuntimeError as e:
        assert mock_task._setup_was_called, "Setup was not called before failure."
        assert mock_task._teardown_was_called, "Teardown was not called after failure"
