"""Copyright © 2026 Pixelgen Technologies AB."""

from pathlib import Path

from pixelator.pna.analysis_engine import AnalysisManager, PerComponentTask
from pixelator.pna.pixeldataset import PixelDatasetSaver, PNAPixelDataset


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
