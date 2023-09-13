"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseAdapterQCTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for adapterqc tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    ```
    """

    __stage_key__ = "adapterqc"

    @pytest.mark.dependency(name="test_adapterqc_run", depends=["test_preqc_run"])
    def test_adapterqc_run(self):
        input_files = (self.workdir / "preqc").glob("*.processed.fastq.gz")

        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        design = self.__get_data("design")

        command = [
            "--log-file",
            "adapterqc-pixelator.log",
            "single-cell",
            "adapterqc",
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
        self.context.run_command("adapterqc", command, input_files)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    def test_adapterqc_logfile_exist(self):
        assert (self.workdir / "adapterqc-pixelator.log").is_file()

    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    def test_adapterqc_results_folder_exists(self):
        assert (self.workdir / "preqc").is_dir()

    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    def test_adapterqc_processed_output_exists(self):
        processed_files = (self.workdir / "adapterqc").glob("*.processed.fastq.gz")
        for f in processed_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    def test_adapterqc_failed_output_exists(self):
        failed_files = (self.workdir / "adapterqc").glob("*.failed.fastq.gz")
        for f in failed_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_adapterqc_run"])
    def test_adapterqc_json_output_exists(self):
        json_files = (self.workdir / "adapterqc").glob("*.report.fastq.gz")
        for f in json_files:
            assert f.is_file()
