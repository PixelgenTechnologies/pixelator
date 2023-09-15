"""Copyright (c) 2023 Pixelgen Technologies AB.

Pytest configuration for integration testing pixelator
"""
import pytest
from types import TracebackType
from typing import Optional, Type

from pixelator.test_utils import YamlIntegrationTestsCollector
from pixelator.test_utils import use_workflow_context  # noqa: F401
from pixelator.types import PathType

import logging
import sys

logger = logging.getLogger("integration")


def handle_unhandled_exception(
    exc_type: Type,
    exc_value: BaseException,
    exc_traceback: Optional[TracebackType],
) -> None:
    """Handle KeyboardInterrupt exceptions.

    Otherwise, raise with `logger.critical`

    :param exc_type: type of the exception
    :param exc_value: the exception exit code
    :param exc_traceback: traceback instance of the exception
    :return: system exception call or `logger.critical`
    :rtype: None
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Will call default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return None

    # Create a critical level log message with info from the except hook.
    logger.critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )
    return None


# Assign the excepthook to the handler
sys.excepthook = handle_unhandled_exception


def pytest_collect_file(
    parent: YamlIntegrationTestsCollector, file_path: PathType
) -> YamlIntegrationTestsCollector:
    """Collect test into a specific instance of :class:`YamlIntegrationTestsCollector`.

    :param parent: the parent object
    :param file_path: path to the yaml file with test definitions
    :return: A custom pytest collector that generates tests from yaml files.
    :rtype: YamlIntegrationTestsCollector
    """
    if file_path.suffix == ".yaml" and file_path.name.startswith("test_"):
        yaml_tests = YamlIntegrationTestsCollector.from_parent(parent, path=file_path)
        yaml_tests.add_marker(pytest.mark.workflow_test)
        return yaml_tests
