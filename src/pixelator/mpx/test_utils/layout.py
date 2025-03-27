"""Copyright © 2023 Pixelgen Technologies AB."""

import logging

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseLayoutTestsMixin(BaseWorkflowTestMixin):
    """Base class for layout command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_layout_run"])
    ```
    """

    __stage_key__ = "layout"

    @pytest.mark.dependency(name="test_layout_run", depends=["test_annotate_run"])
    def test_layout_run(self):
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        input_files = list(
            map(
                lambda f: str(f),
                (self.workdir / "annotate").glob("*.annotate.dataset.pxl"),
            )
        )

        command = [
            "--log-file",
            "layout-pixelator.log",
            "single-cell-mpx",
            "layout",
            "--output",
            ".",
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        self.context.run_command("layout", command, input_files)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_layout_run"])
    def test_layout_dataset_exists(self):
        pxl_files = (self.workdir / "layout").glob("*.layout.dataset.pxl")
        for f in pxl_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_layout_run"])
    def test_layout_report_exists(self):
        json_files = (self.workdir / "layout").glob("*.report.json")
        for f in json_files:
            assert f.is_file()
