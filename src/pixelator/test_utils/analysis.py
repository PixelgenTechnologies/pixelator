"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseAnalysisTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for analysis command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_analysis_run"])
    ```
    """

    __stage_key__ = "analysis"

    @pytest.mark.dependency(name="test_analysis_run", depends=["test_annotate_run"])
    def test_analysis_run(self):
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        input_files = list(
            map(lambda f: str(f), (self.workdir / "annotate").glob("*.dataset.pxl"))
        )

        command = [
            "--log-file",
            "analysis-pixelator.log",
            "single-cell",
            "analysis",
            "--output",
            ".",
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        self.context.run_command("analysis", command, input_files)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_analysis_run"])
    def test_analysis_dataset_exists(self):
        pxl_files = (self.workdir / "analysis").glob("*.dataset.pxl")
        for f in pxl_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_analysis_run"])
    def test_analysis_report_exists(self):
        json_files = (self.workdir / "analysis").glob("*.report.json")
        for f in json_files:
            assert f.is_file()
