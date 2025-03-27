"""Workflow test helper for single-cell report command.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging
from pathlib import Path

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseReportTestsMixin(BaseWorkflowTestMixin):
    """Base class for report command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_report_run"])
    ```
    """

    __stage_key__ = "report"

    @pytest.mark.dependency(name="test_report_run", depends=["test_analysis_run"])
    def test_report_run(self):
        """Test and run the report command."""
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        panel = self.__get_data("panel")
        panel_file = self.__get_data("panel_file")

        command = [
            "--log-file",
            "report-pixelator.log",
            "single-cell-mpx",
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
    def test_report_qc_report_exists(self):
        """Check if the qc report html exists."""
        report_files = list((self.workdir / "report").glob("*.qc-report.html"))
        assert len(report_files) > 0
