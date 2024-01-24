"""Logging setup for pixelator.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import atexit
import functools
import logging
import logging.handlers
import multiprocessing
import sys
import warnings
from pathlib import Path
from typing import Callable

import click
from numba import NumbaDeprecationWarning

from pixelator.utils import click_echo

root_logger = logging.getLogger()
pixelator_root_logger = logging.getLogger("pixelator")


# Silence deprecation warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="geopandas not available. Some functionality will be disabled.",
    category=UserWarning,
)


# Disable matplot lib debug logs (that will clog all debug logs)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.ticker").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)


# ------------------------------------------------------------
# Click logging
# ------------------------------------------------------------

LOGGER_KEY = __name__ + ".logger"
DEFAULT_LEVEL = logging.INFO


class ColorFormatter(logging.Formatter):
    """Click formatter with colored levels"""

    colors = {
        "debug": dict(fg="blue"),
        "info": dict(fg="green"),
        "warning": dict(fg="yellow"),
        "error": dict(fg="orange"),
        "exception": dict(fg="red"),
        "critical": dict(fg="red"),
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with colored level.

        :param record: The record to format.
        :returns: A formatted log record.
        """
        if not record.exc_info:
            level = record.levelname.lower()
            msg = record.getMessage()
            if level in self.colors:
                timestamp = self.formatTime(record, self.datefmt)
                colored_level = click.style(
                    f"{level.upper():<10}", **self.colors[level]
                )
                prefix = f"{timestamp} [{colored_level}]  "
                msg = "\n".join(prefix + x for x in msg.splitlines())
            return msg
        return logging.Formatter.format(self, record)


class DefaultCliFormatter(ColorFormatter):
    """Click formatter with colored levels"""

    def format(self, record):
        """Format a record for CLI output."""
        if not record.exc_info:
            level = record.levelname.lower()
            msg = record.getMessage()

            if level == "info":
                return msg

            return f"{level.upper()}: {msg}"
        return logging.Formatter.format(self, record)


class ClickHandler(logging.Handler):
    """Click logging handler.

    Messages are forwarded to stdout using `click.echo`.
    """

    _use_stderr = True

    def emit(self, record):
        """
        Do whatever it takes to actually log the specified logging record.

        This version is intended to be implemented by subclasses and so
        raises a NotImplementedError.
        """
        try:
            msg = self.format(record)
            click.echo(msg, err=self._use_stderr)
        except Exception:
            self.handleError(record)


class LoggingSetup:
    """Logging setup for multiprocessing.

    This class is used to set up logging for multiprocessing.
    All messages are passed to a separate process that handles the logging to a file.


    """

    def __init__(self, log_file: Path, verbose: bool, logger=None):
        """Initialize the logging setup.

        :param log_file: the filename of the log output
        :param verbose: enable verbose logging and console output
        :param logger: the logger to configure, default is the root logger
        """
        self.log_file = log_file
        self.verbose = verbose
        self._root_logger = logger or logging.getLogger()
        self._queue: multiprocessing.Queue = multiprocessing.Queue(-1)

        self._listener_process = multiprocessing.Process(
            name="log-process",
            target=self._listener_process_main,
            args=(
                self._queue,
                functools.partial(
                    self._listener_logging_setup, self.log_file, self.verbose
                ),
            ),
        )
        atexit.register(self._shutdown_listener, self._queue, self._listener_process)

    @staticmethod
    def _shutdown_listener(queue, process):
        """Send the stop token to the logging process and wait for it to finish."""
        queue.put(None)
        process.join()

    def initialize(self):
        """Configure logging and start the listener process."""
        self._listener_process.start()

        handler = logging.handlers.QueueHandler(self._queue)
        self._root_logger.handlers = [handler]
        self._root_logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        console_handler = ClickHandler()
        if not self.verbose:
            console_handler.setFormatter(DefaultCliFormatter())
            self._root_logger.addHandler(console_handler)
        else:
            console_handler.setFormatter(ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
            self._root_logger.addHandler(console_handler)

    @staticmethod
    def _listener_logging_setup(log_file: str, verbose: bool):
        """Initialize the logging in the listener process."""
        logger = logging.getLogger()
        handler = logging.FileHandler(str(log_file), mode="w")
        formatter = logging.Formatter(
            "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    @staticmethod
    def _listener_process_main(
        queue: multiprocessing.Queue, configure: Callable[[], None]
    ):
        """Entrypoint for the logging process.

        :param queue: the queue to read log messages from
        :param configure: the function to call to configure the logging
        """
        configure()

        logging.info("Logging process started.")
        while True:
            try:
                record = queue.get()
                if (
                    record is None
                ):  # We send this as a sentinel to tell the listener to quit.
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)  # No level or filter logic applied - just do it!
            except Exception:
                import traceback

                click_echo("An unexpected error occurred in the log handler")
                click_echo(traceback.format_exc())

        logging.info("Logging process stopped.")


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    """Handle "unhandled" exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Will call default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Create a critical level log message with info from the except hook.
    pixelator_root_logger.critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# Assign the excepthook to the handler
sys.excepthook = handle_unhandled_exception
