"""Logging setup for pixelator.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import atexit
import functools
import logging
import logging.handlers
import multiprocessing
import sys
import time
import typing
import warnings
from pathlib import Path
from typing import Callable

import click
from numba import NumbaDeprecationWarning

from pixelator.types import PathType
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


class StyleDict(typing.TypedDict):
    """Style dictionary for kwargs to `click.style`."""

    fg: str


class ColorFormatter(logging.Formatter):
    """Click formatter with colored levels"""

    colors: dict[str, StyleDict] = {
        "debug": StyleDict(fg="blue"),
        "info": StyleDict(fg="green"),
        "warning": StyleDict(fg="yellow"),
        "error": StyleDict(fg="red"),
        "exception": StyleDict(fg="red"),
        "critical": StyleDict(fg="red"),
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with colored level.

        :param record: The record to format.
        :returns str: A formatted log record.
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


class DefaultCliFormatter(logging.Formatter):
    """Click formatter with colored levels"""

    def format(self, record: logging.LogRecord) -> str:
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

    :param use_stderr: Log to sys.stderr instead of sys.stdout.
    """

    def __init__(self, use_stderr: bool = True):
        """Initialize the click handler."""
        self._use_stderr = use_stderr

    def emit(self, record: logging.LogRecord) -> None:
        """Do whatever it takes to actually log the specified logging record."""
        try:
            msg = self.format(record)
            click.echo(msg, file=sys.stdout, err=self._use_stderr)
        except Exception:
            self.handleError(record)


class LoggingSetup:
    """Logging setup for multiprocessing.

    This class is used to set up logging for multiprocessing.
    All messages are passed to a separate process that handles the logging to a file.
    """

    def __init__(self, log_file: PathType | None, verbose: bool, logger=None):
        """Initialize the logging setup.

        :param log_file: the filename of the log output
        :param verbose: enable verbose logging and console output
        :param logger: the logger to configure, default is the root logger
        """
        self.log_file = Path(log_file) if log_file is not None else None
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

    def _shutdown_listener(self, timeout: int = 10) -> None:
        """Send the stop token to the logging process and wait for it to finish.

        :param timeout: The number of seconds to wait for the process to finish.
        """
        self._queue.put(None)

        start = time.time()
        while time.time() - start <= timeout:
            if self._listener_process.is_alive():
                time.sleep(0.1)
            else:
                # we cannot join a process that is not started
                if self._listener_process.ident is not None:
                    self._listener_process.join()
                break
        else:
            # We only enter this if we didn't 'break' above during the while loop!
            self._listener_process.terminate()

    def initialize(self):
        """Configure logging and start the listener process."""
        # We do not need the listener process if there is no file to log to
        if self.log_file:
            self._listener_process.start()

        handlers = []
        handlers.append(logging.handlers.QueueHandler(self._queue))
        self._root_logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        console_handler = ClickHandler()
        if self.verbose:
            console_handler.setFormatter(ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            console_handler.setFormatter(DefaultCliFormatter())

        handlers.append(console_handler)
        self._root_logger.handlers = handlers
        atexit.register(self._shutdown_listener)

    def __enter__(self):
        """Enter the context manager.

        This will initialize the logging setup.
        """
        self.initialize()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager.

        This will shut down the logging process if needed.
        """
        self._shutdown_listener()
        atexit.unregister(self._shutdown_listener)
        # Reraise exception higher up the stack
        return False

    @staticmethod
    def _listener_logging_setup(log_file: Path | None, verbose: bool):
        """Initialize the logging in the listener process."""
        logger = logging.getLogger()

        if log_file:
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
