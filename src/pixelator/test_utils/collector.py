"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.test_utils import PixelatorWorkflowTest, WorkflowConfig


class YamlIntegrationTestsCollector(pytest.File):
    """
    Custom pytest collector generating tests from yaml files.
    """

    def collect(self, parent_module=None):
        # We need a yaml parser, e.g. PyYAML.
        config = WorkflowConfig(self.path)

        for case in config.keys():
            name = case.split("-")[1]
            wf = type(
                f"TestSmallWorkflow{name}",
                (PixelatorWorkflowTest,),
                {"test_id": case, **config.get_test_config(case)},
            )

            collector = pytest.Class.from_parent(parent=self, name=name)
            # Need to add this explicitly since the constructor does not set obj
            collector.obj = wf

            for t in collector.collect():
                if t.parent is None:
                    t.parent = self
                t.fixturenames.append("use_workflow_context")
                yield t
