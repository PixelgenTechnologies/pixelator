"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
from pathlib import Path

import pytest

from pixelator.config import config
from pixelator.config.panel import load_antibody_panel
from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseDemuxTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for demux command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_demux_run"])
    ```
    """

    __stage_key__ = "demux"

    @pytest.mark.dependency(name="test_demux_run", depends=["test_adapterqc_run"])
    def test_demux_run(self):
        design = self.__get_data("design")
        panel = self.__get_data("panel")
        panel_file = self.__get_data("panel_file")
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        input_files = list(
            map(
                lambda f: str(f),
                (self.workdir / "adapterqc").glob("*.processed.fastq.gz"),
            )
        )

        command = [
            "--log-file",
            "demux-pixelator.log",
            "single-cell",
            "demux",
            "--panel",
            str(Path(panel_file).name) if panel_file else panel,
            "--output",
            ".",
            "--design",
            design,
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        file_deps = list(input_files)
        if panel_file is not None:
            file_deps.append(panel_file)

        self.context.run_command("demux", command, file_deps)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_demux_run"])
    def test_demux_logfile_exist(self):
        assert (self.workdir / "adapterqc-pixelator.log").is_file()

    @pytest.mark.dependency(depends=["test_demux_run"])
    def test_demux_results_folder_exists(self):
        assert (self.workdir / "preqc").is_dir()

    @pytest.mark.dependency(depends=["test_demux_run"])
    def test_demux_processed_output_exists(self):
        panel_key_or_file = self.__get_data("panel") or self.__get_options(
            "common"
        ).get("panel_file")
        panel = load_antibody_panel(config, panel_key_or_file)

        for marker in panel.markers:
            processed_files = (self.workdir / "demux").glob(
                f"*.processed-{marker}.fastq.gz"
            )
            for f in processed_files:
                assert f.is_file()

    @pytest.mark.dependency(depends=["test_demux_run"])
    def test_demux_failed_output_exists(self):
        failed_files = (self.workdir / "demux").glob("*.failed.fastq.gz")
        for f in failed_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_demux_run"])
    def test_demux_json_output_exists(self):
        json_files = (self.workdir / "demux").glob("*.report.json")
        for f in json_files:
            assert f.is_file()
