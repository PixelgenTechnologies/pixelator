"""Annotation tests for the graph step.

Copyright © 2023 Pixelgen Technologies AB.
"""

import logging

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseGraphTestsMixin(BaseWorkflowTestMixin):
    """Base class for graph command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_graph_run"])
    ```
    """

    __stage_key__ = "graph"

    @pytest.mark.dependency(name="test_graph_run", depends=["test_collapse_run"])
    def test_graph_run(self):
        """Run the graph command."""
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")

        input_files = list(
            map(
                lambda f: str(f),
                (self.workdir / "collapse").glob("*.collapsed.parquet"),
            )
        )

        command = [
            "--log-file",
            "graph-pixelator.log",
            "single-cell-mpx",
            "graph",
            "--output",
            ".",
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        self.context.run_command("graph", command, input_files)
        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_graph_run"])
    def test_graph_edgelist_exists(self):
        """Check that the edgelist exists."""
        edge_files = (self.workdir / "graph").glob("*.edgelist.parquet")
        for f in edge_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_graph_run"])
    def test_graph_report_exists(self):
        """Check that the json report exists."""
        json_files = (self.workdir / "graph").glob("*.report.json")
        for f in json_files:
            assert f.is_file()
