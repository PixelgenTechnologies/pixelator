"""Custom pytest collector utilities to generate tests from yaml files.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from .config import WorkflowConfig
from .workflow import PixelatorWorkflowTest


class NoModuleClass(pytest.Class):  # noqa
    """Mock pytest.Class that does not require a module as a parent."""

    class MockModule:
        """Fake module with a None obj."""

        def __init__(self):  # noqa
            self.obj = None

    def getparent(self, cls):  # noqa
        parent = super().getparent(cls)
        if parent is None:
            return self.MockModule()
        return parent


class YamlIntegrationTestsCollector(pytest.File):
    """Custom pytest collector generating tests from yaml files."""

    def collect(self):
        """Convert yaml files to test cases."""
        config = WorkflowConfig(self.path)

        for case in config.keys():
            name = case.split("-")[1]
            case_config = config.get_test_config(case)
            wf = type(
                f"TestWorkflow{name}",
                (PixelatorWorkflowTest,),
                {"test_id": case, **case_config},
            )

            collector = NoModuleClass.from_parent(parent=self, name=self.path.stem)
            # Need to add this explicitly since the constructor does not set obj
            collector.obj = wf

            for t in collector.collect():
                if t.parent is None:
                    t.parent = self
                t.fixturenames.append("use_workflow_context")
                yield t
