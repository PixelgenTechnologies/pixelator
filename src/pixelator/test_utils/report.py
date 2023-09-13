"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
from pathlib import Path

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseReportTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for report command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_report_run"])
    ```
    """

    __stage_key__ = "report"

    @pytest.mark.dependency(name="test_report_run", depends=["test_analysis_run"])
    def test_report_run(self):
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        panel = self.__get_data("panel")
        panel_file = self.__get_data("panel_file")

        command = [
            "--log-file",
            "report-pixelator.log",
            "single-cell",
            "report",
            ".",
            "--output",
            ".",
            "--panel",
            str(Path(panel_file).name) if panel_file else panel,
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        file_deps = []
        if panel_file is not None:
            file_deps.append(panel_file)

        self.context.run_command("report", command, file_deps)

        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_report_run"])
    def test_report_customer_exists(self):
        report_files = (self.workdir / "report").glob("*_report.html")
        for f in report_files:
            assert f.is_file()
