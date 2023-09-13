"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
from pathlib import Path

import pytest

from .base import BaseWorkflowTestMixin

logger = logging.getLogger(__name__)


class BaseAnnotateTestsMixin(BaseWorkflowTestMixin):
    """
    Base class for annotate command tests.

    Test cases (defined in this class or in subclasses)
    that depend on the output should be marked with:
    ```
    @pytest.mark.dependency(depends=["test_annotate_run"])
    ```
    """

    __stage_key__ = "annotate"

    @pytest.mark.dependency(name="test_annotate_run", depends=["test_graph_run"])
    def test_annotate_run(self):
        params = self.__get_parameters()
        verbose = self.__get_options("common").get("verbose")
        panel = self.__get_data("panel")
        panel_file = self.__get_data("panel_file")

        input_files = list(
            map(lambda f: str(f), (self.workdir / "graph").glob("*.edgelist.csv.gz"))
        )

        command = [
            "--log-file",
            "annotate-pixelator.log",
            "single-cell",
            "annotate",
            "--panel",
            str(Path(panel_file).name) if panel_file else panel,
            "--output",
            ".",
        ]

        if verbose:
            command = ["--verbose"] + command

        if params:
            command += params

        command += input_files

        file_deps = list(input_files)
        if panel_file is not None:
            file_deps.append(panel_file)

        self.context.run_command("annotate", command, file_deps)
        with open(str(self.workdir / "annotate-pixelator.log")) as f:
            self.context.set_logs("annotate", f.read())

        assert self.__this_result.exit_code == 0

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_annotate_dataset_exists(self):
        dataset_files = (self.workdir / "annotate").glob("*.dataset.pxl")
        for f in dataset_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_annotate_report_exists(self):
        json_files = (self.workdir / "annotate").glob("*.report.json")
        for f in json_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_annotate_images_exist(self):
        png_files = (self.workdir / "annotate").glob("*.rank_vs_size.png")

        if "--verbose" not in self.__this_command:
            return pytest.skip(
                'Skipping rank_vs_size image test, ( "--verbose" not set )'
            )

        for f in png_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_raw_component_metrics_exist(self):
        metric_files = (self.workdir / "annotate").glob(
            "*.raw_components_metrics.csv.gz"
        )

        if "--verbose" not in self.__this_command:
            return pytest.skip(
                'Skipping raw_components_metrics.csv.gz test, ( "--verbose" not set )'
            )

        for f in metric_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_doublet_calling_vertex_communities_exists(self):
        vertex_files = (self.workdir / "annotate").glob("*.vertex_communities.csv.gz")

        if "--doublet-calling" not in self.__this_command:
            return pytest.skip(
                "Skipping vertex_communities.csv.gz test, "
                '( "--doublet-calling" not set )'
            )

        if "--verbose" not in self.__this_command:
            return pytest.skip(
                "Skipping raw_components_metrics.csv.gz test, "
                '( "--verbose" not set )'
            )

        for f in vertex_files:
            assert f.is_file()

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_umap_png_exists(self):
        png_files = (self.workdir / "annotate").glob("*.umap.png")

        if "--verbose" not in self.__this_command:
            return pytest.skip('Skipping umap.png test, ( "--verbose" not set )')

        if "Skipping clustering" in self.__this_logs:
            return pytest.skip(
                'Skipping umap.png test, ( "Skipping clustering" in output )'
            )

        for f in png_files:
            assert f.is_file()
