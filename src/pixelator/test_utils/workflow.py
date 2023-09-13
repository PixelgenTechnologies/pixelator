"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
from pathlib import Path

from . import (
    BaseAdapterQCTestsMixin,
    BaseAmpliconTestsMixin,
    BaseAnalysisTestsMixin,
    BaseAnnotateTestsMixin,
    BaseCollapseTestsMixin,
    BaseDemuxTestsMixin,
    BaseGraphTestsMixin,
    BasePreQCTestsMixin,
    BaseReportTestsMixin,
)
from .workflow_context import PixelatorWorkflowContext


# IMPORTANT: Note the order of the base classes.
# This is important to ensure that dependencies are resolved correctly.
# A mixin with dependencies on test cases defined in other mixins must be
# listed **before** the dependent mixins.
class PixelatorWorkflowTest(
    BaseReportTestsMixin,
    BaseAnalysisTestsMixin,
    BaseAnnotateTestsMixin,
    BaseGraphTestsMixin,
    BaseCollapseTestsMixin,
    BaseDemuxTestsMixin,
    BaseAdapterQCTestsMixin,
    BasePreQCTestsMixin,
    BaseAmpliconTestsMixin,
):
    """
    Pixelator workflow test base class.

    This class bundles all the basic tests for each command in the pixelator workflow.
    It is intended to be used as a base class for new workflow tests.
    New tests cases per command can be added by creating a new method and marking it
    with a dependency decorator. The command output of each stage is called
    test_<stage name>_run.

    So for a test case testing annotate output:

    @pytest.mark.dependency(depends=["test_annotate_run"])
    def test_some_annotate_output(self):
        pass

    The workflow test is configurable by setting class variables in the subclass.
    The current mixin plugins are defined to use the name of the stage they are
    testing, eg. the basic collapse tests from :class:`BaseCollapseTestsMixin` will
    use the `collapse` dictionary from the subclass. The dictionary can be contain
    arbitrary values for configurating the tests.

    The basic mixins use `params` defined in the dict for each subcommand
    to pass arbitrary commandline arguments.

    When a key is not found in the dictionary, the test will look for a key
    in the test configuration yaml file. This is used in the mixin class for eg.
    `design` and `sample_ids` which are across multiple mixin classes.
    """

    # This will be injected by a fixture
    context: PixelatorWorkflowContext

    @property
    def workdir(self) -> Path:
        """
        Return the workdir from the workflow context.
        """
        return self.context.workdir
