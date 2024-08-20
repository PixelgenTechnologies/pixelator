"""Tests for the pixelator logging module.

Copyright © 2024 Pixelgen Technologies AB.
"""

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from pixelator.logging import LoggingSetup
from pixelator.utils import (
    get_process_pool_executor,
    timer,
)


def test_timer(caplog):
    @timer
    def my_func():
        return "foo"

    with caplog.at_level(logging.INFO):
        res = my_func()
        assert res == "foo"
        assert "Finished pixelator my_func in" in caplog.text


def test_verbose_logging_is_activated():
    test_log_file = tempfile.NamedTemporaryFile()
    with LoggingSetup(test_log_file.name, verbose=True):
        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.DEBUG


def test_verbose_logging_is_deactivated():
    test_log_file = tempfile.NamedTemporaryFile()
    with LoggingSetup(test_log_file.name, verbose=False):
        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO


def assert_messages_eventually_in_file(messages, log_file, tries=3, timeout=0.5):
    try:
        with open(log_file.name, "r") as f:
            log_content = f.read()
            for message in messages:
                assert message in log_content
    except AssertionError as e:
        if tries > 0:
            time.sleep(timeout)
            assert_messages_eventually_in_file(
                messages, log_file, tries=tries - 1, timeout=timeout * 1.5
            )
        raise e


def helper_log_fn(args):
    import logging

    root_logger = logging.getLogger()
    root_logger.log(*args)


@pytest.mark.parametrize("verbose", [True, False])
def test_single_processes_logging(verbose):
    """Test that logging works in a single process environment."""
    test_log_file = tempfile.NamedTemporaryFile()

    with LoggingSetup(test_log_file.name, verbose=verbose):
        tasks = [
            (logging.DEBUG, "This is a debug message"),
            (logging.INFO, "This is an info message"),
            (logging.WARNING, "This is a warning message"),
            (logging.ERROR, "This is an error message"),
            (logging.CRITICAL, "This is a critical message"),
        ]

        for task in tasks:
            helper_log_fn(task)

    # Test that the console output has the expected messages
    messages = [
        "This is an info message",
        "This is a warning message",
        "This is an error message",
        "This is a critical message",
    ]

    if verbose:
        messages.append("This is a debug message")

    assert_messages_eventually_in_file(messages, test_log_file)
    if not verbose:
        assert "This is a debug message" not in Path(test_log_file.name).read_text()


@pytest.mark.parametrize("verbose", [True, False])
def test_single_processes_logging_no_context_handler(verbose):
    """Test that logging works in a single process environment."""
    test_log_file = tempfile.NamedTemporaryFile()

    logging_setup = LoggingSetup(test_log_file.name, verbose=verbose)
    logging_setup.initialize()

    tasks = [
        (logging.DEBUG, "This is a debug message"),
        (logging.INFO, "This is an info message"),
        (logging.WARNING, "This is a warning message"),
        (logging.ERROR, "This is an error message"),
        (logging.CRITICAL, "This is a critical message"),
    ]

    for task in tasks:
        helper_log_fn(task)

    # Test that the console output has the expected messages
    messages = [
        "This is an info message",
        "This is a warning message",
        "This is an error message",
        "This is a critical message",
    ]

    if verbose:
        messages.append("This is a debug message")

    # Closing the logging setup flushes it.
    logging_setup.close()

    assert_messages_eventually_in_file(messages, test_log_file)
    if not verbose:
        assert "This is a debug message" not in Path(test_log_file.name).read_text()


@pytest.mark.parametrize("verbose", [True, False])
def test_multiprocess_logging(verbose):
    """Test that logging works in a multiprocess environment."""
    test_log_file = tempfile.NamedTemporaryFile()

    with LoggingSetup(test_log_file.name, verbose=verbose) as logging_setup:
        tasks = [
            (logging.DEBUG, "This is a debug message"),
            (logging.INFO, "This is an info message"),
            (logging.WARNING, "This is a warning message"),
            (logging.ERROR, "This is an error message"),
            (logging.CRITICAL, "This is a critical message"),
        ]
        # Note that for most cases we should use context="spawn" (default) as this is
        # safe while fork is not. However, using "fork" here considerably speeds up the
        # tests.
        with get_process_pool_executor(
            nbr_cores=4, logging_setup=logging_setup, context="fork"
        ) as executor:
            for r in executor.map(helper_log_fn, tasks):
                pass

    # Test that the console output has the expected messages
    messages = [
        "This is an info message",
        "This is a warning message",
        "This is an error message",
        "This is a critical message",
    ]

    if verbose:
        messages.append("This is a debug message")

    assert_messages_eventually_in_file(messages, test_log_file)
    if not verbose:
        assert "This is a debug message" not in Path(test_log_file.name).read_text()


def raise_unhandled_exception():
    raise ValueError("This is an unhandled exception")


def test_unhandled_exception(caplog):
    """Test that raising an unexpected exception gets added to the log"""

    with pytest.raises(ValueError, match="This is an unhandled exception"):
        with LoggingSetup():
            raise ValueError("This is an unhandled exception")

    assert "Unhandled exception of type: ValueError" in caplog.messages
    assert "Exception message was: This is an unhandled exception" in caplog.messages
    assert "Traceback" in caplog.text


def test_unhandled_exception_multiprocess_logging(caplog):
    """Test that raising an unexpected exception gets added to the log when multiprocessing"""

    with pytest.raises(ValueError, match="This is an unhandled exception"):
        with LoggingSetup() as logging_setup:
            with get_process_pool_executor(
                nbr_cores=4, logging_setup=logging_setup, context="fork"
            ) as executor:
                executor.submit(raise_unhandled_exception).result()

    assert "Unhandled exception of type: ValueError" in caplog.messages
    assert "Exception message was: This is an unhandled exception" in caplog.messages
    assert "Traceback" in caplog.text


def test_unhandled_exception_multithreading_logging(caplog):
    """Test that raising an unexpected exception gets added to the log when multithreading"""

    with pytest.raises(ValueError, match="This is an unhandled exception"):
        with LoggingSetup():
            with ThreadPoolExecutor(max_workers=4) as executor:
                executor.submit(raise_unhandled_exception).result()

    assert "Unhandled exception of type: ValueError" in caplog.messages
    assert "Exception message was: This is an unhandled exception" in caplog.messages
    assert "Traceback" in caplog.text
