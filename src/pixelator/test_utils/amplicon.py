"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
from pathlib import Path

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseAmpliconTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for amplicon command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_amplicon_run"])
    ```
    """

    __stage_key__ = "amplicon"

    @pytest.mark.dependency(name="test_amplicon_run")
    def test_amplicon_run(self):
        input_files = self.__get_data("input_files")
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        design = self.__get_data("design")

        command = [
            "--log-file",
            "amplicon-pixelator.log",
            "single-cell",
            "amplicon",
            "--output",
            ".",
            "--design",
            str(design),
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += [Path(s).name for s in input_files]

        self.context.run_command("amplicon", command, input_files)

        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_amplicon_run"])
    def test_amplicon_logfile_exist(self):
        assert (self.workdir / "amplicon-pixelator.log").exists()

    @pytest.mark.dependency(depends=["test_amplicon_run"])
    def test_amplicon_results_folder_exists(self):
        assert (self.workdir / "amplicon").exists()

    @pytest.mark.dependency(depends=["test_amplicon_run"])
    def test_amplicon_results_file_exists(self):
        exists_files = (self.workdir / "amplicon").glob("*.merged.fastq.gz")
        for f in exists_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_amplicon_run"])
    def test_amplicon_results_file_not_empty(self):
        not_empty_files = (self.workdir / "amplicon").glob("*.merged.fastq.gz")
        for f in not_empty_files:
            assert f.stat().st_size > 0
