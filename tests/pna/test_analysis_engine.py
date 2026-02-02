from pathlib import Path

from pixelator.pna.analysis_engine import AnalysisManager, PerComponentTask
from pixelator.pna.pixeldataset import PixelDatasetSaver, PNAPixelDataset


class FailingTask(PerComponentTask):
    """A task that simulates failure during execution."""

    TASK_NAME = "failing_task"

    def __init__(self, work_folder: Path | None = None):
        super().__init__()
        self._work_folder = work_folder

    def run_from_component_id(self, component_id: str):
        raise RuntimeError("Simulated failure")


def test_tempfile_cleanup_on_failure(pna_pxl_dataset: PNAPixelDataset, tmp_path):
    manager = AnalysisManager(
        [
            FailingTask(),
        ]
    )
    pna_pixel_filtered = pna_pxl_dataset.filter(
        components=pna_pxl_dataset.adata().obs.index[:2]
    )

    pxl_file_target = PixelDatasetSaver(pxl_dataset=pna_pixel_filtered).save(
        "PNA055_Sample07_S7", Path(tmp_path) / "layout.pxl"
    )
    try:
        dataset = manager.execute_from_path(pxl_file_target.path, pxl_file_target)
    except RuntimeError as e:
        assert str(e) == "Simulated failure", "Unexpected error message."
        assert manager._temp_folders_used, "No temporary folders were used."
        for temp_folder in manager._temp_folders_used:
            assert not temp_folder.exists(), (
                f"Temporary folder {temp_folder} was not cleaned up."
            )
