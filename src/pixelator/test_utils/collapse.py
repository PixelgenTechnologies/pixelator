"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
from pathlib import Path

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseCollapseTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for collapse command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_collapse_run"])
    ```
    """

    __stage_key__ = "collapse"

    @pytest.mark.dependency(name="test_collapse_run", depends=["test_demux_run"])
    def test_collapse_run(self):
        input_files = [
            str(f) for f in (self.workdir / "demux").glob("*processed*fastq.gz")
        ]
        panel = self.__get_data("panel")
        panel_file = self.__get_data("panel_file")
        design = self.__get_data("design")
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")

        command = [
            "--log-file",
            "collapse-pixelator.log",
            "single-cell",
            "collapse",
            "--output",
            ".",
            "--design",
            design,
            "--panel",
            str(Path(panel_file).name) if panel_file else panel,
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        file_deps = list(input_files)
        if panel_file is not None:
            file_deps.append(panel_file)

        self.context.run_command("collapse", command, file_deps)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_collapse_run"])
    def test_collapse_logfile_exist(self):
        assert (self.workdir / "collapse-pixelator.log").is_file()

    @pytest.mark.dependency(depends=["test_collapse_run"])
    def test_collapse_results_folder_exists(self):
        assert (self.workdir / "collapse").is_dir()
