"""Workflow test helper for single-cell preqc command.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BasePreQCTestsMixin(BaseWorkflowTestMixin):
    """Base class for preqc command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_preqc_run"])
    ```
    """

    __stage_key__ = "preqc"

    @pytest.mark.dependency(name="test_preqc_run", depends=["test_amplicon_run"])
    def test_preqc_run(self):
        """Test and run the preqc command."""
        design = self.__get_data("design")
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")

        input_files = list(
            map(
                lambda f: str(f),
                (self.workdir / "amplicon").glob("*.merged.fastq.gz"),
            )
        )

        command = [
            "--log-file",
            "preqc-pixelator.log",
            "single-cell-mpx",
            "preqc",
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

        self.context.run_command("preqc", command, input_files)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_logfile_exist(self):
        """Check if the log file exists."""
        assert (self.workdir / "preqc-pixelator.log").is_file()

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_results_folder_exists(self):
        """Check if the output directory is created."""
        assert (self.workdir / "preqc").is_dir()

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_processed_output_exists(self):
        """Check if the per antibody "processed" fastq files exist."""
        processed_files = (self.workdir / "preqc").glob("*.processed.fastq.gz")
        for f in processed_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_failed_output_exists(self):
        """Check if the "failed" fastq files exist."""
        failed_files = (self.workdir / "preqc").glob("*.failed.fastq.gz")
        for f in failed_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_html_output_exists(self):
        """Check if the fastp html report exist."""
        html_files = (self.workdir / "preqc").glob("*.report.html")
        for f in html_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_preqc_run"])
    def test_preqc_json_output_exists(self):
        """Check if the report.json summary exist."""
        json_files = (self.workdir / "preqc").glob("*.report.json")
        for f in json_files:
            assert f.is_file()
