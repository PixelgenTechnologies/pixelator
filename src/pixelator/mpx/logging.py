"""Logging setup for pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
import pickle
import socketserver
import struct
import sys
import threading
import time
import traceback
import typing
import warnings
from logging.handlers import DEFAULT_TCP_LOGGING_PORT, SocketHandler
from pathlib import Path

import click
from numba import NumbaDeprecationWarning

from pixelator.common.types import PathType

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


class StyleDict(typing.TypedDict):
    """Style dictionary for kwargs to `click.style`."""

    fg: str


class ColorFormatter(logging.Formatter):
    """Click formatter with colored levels."""

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
    """Click formatter with colored levels."""

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
    """

    def __init__(self, level: int = 0, use_stderr: bool = True):
        """Initialize the click handler.

        :param level: The logging level.
        :param use_stderr: Log to sys.stderr instead of sys.stdout.
        """
        super().__init__(level=level)
        self._use_stderr = use_stderr

    def emit(self, record: logging.LogRecord) -> None:
        """Do whatever it takes to actually log the specified logging record.

        :param record: The record to log.
        """
        try:
            msg = self.format(record)
            click.echo(msg, err=self._use_stderr)
        except Exception:
            self.handleError(record)


LOCALHOST = "localhost"


class LoggingSetup:
    """Logging setup for pixelator.

    This class is used to set up logging for pixelator. The reason we need this somewhat
    involved setup is to be able to deal with

    All messages are passed to a separate process that handles the logging to a file.
    """

    def __init__(
        self, log_file: PathType | None = None, verbose: bool = False, logger=None
    ):
        """Initialize the logging setup.

        :param log_file: the filename of the log output
        :param verbose: enable verbose logging and console output
        :param logger: the logger to configure, default is the root logger
        """
        self.log_file = Path(log_file) if log_file is not None else None
        self.verbose = verbose
        self._root_logger = logger or logging.getLogger()
        self._server = None
        self._server_thread = None

    def _shutdown_listener(self) -> None:
        """Send the stop token to the logging process and wait for it to finish."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread:
            self._server_thread.join(timeout=1)

    def initialize(self):
        """Configure logging and start the listener process."""
        self._server = LogRecordSocketReceiver(
            host=LOCALHOST,
            # Setting the port to 0 asks the OS to assign a free port
            port=0,
            handler=LogRecordStreamHandler,
            log_file=self.log_file,
            console_log_formatter=(
                ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
                if self.verbose
                else DefaultCliFormatter()
            ),
        )
        _, port = self._server.server_address
        self.port = port
        self._server_thread = threading.Thread(
            target=self._server.serve_forever, name="pixelator-log-listener"
        )
        self._server_thread.daemon = True
        self._server_thread.start()

        socker_handler = SocketHandler(LOCALHOST, port)
        self._root_logger.addHandler(socker_handler)
        self._root_logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

    @property
    def log_level(self):
        """Return the current log level."""
        return self._root_logger.level

    def __enter__(self):
        """Enter the context manager.

        This will initialize the logging setup.
        """
        self.initialize()
        return self

    def close(self):
        """Close down the logging setup."""
        self._shutdown_listener()

    def __exit__(self, exc_type, exc_value, traceback_obj):
        """Exit the context manager.

        This will shut down the logging process if needed.
        """

        def log_exception(exc_type, exc_value, traceback_obj):
            if issubclass(exc_type, click.exceptions.UsageError) or issubclass(
                exc_type, click.exceptions.Exit
            ):
                # Click exceptions are dealt with separately
                # and thus we ignore them here.
                return False

            if issubclass(exc_type, SystemExit):
                # SystemExit is raised when the application has been explicitly
                # directed to exit, so we don't what a trace dumped for that.
                return False

            self._root_logger.critical(
                "Unhandled exception of type: {}".format(exc_type.__name__)
            )
            self._root_logger.critical("Exception message was: {}".format(exc_value))
            for item in traceback.format_exception(exc_type, exc_value, traceback_obj):
                for line in item.splitlines():
                    self._root_logger.critical(line)

        try:
            # It seems that sometimes the current exception is not passed
            # to the __exit__ method. This allows us to check if there
            # is any raised exception and make sure this gets added to
            # the log.
            if exc_type is None:
                exc_type, exc_value, traceback_obj = sys.exc_info()
                if exc_type is not None:
                    log_exception(exc_type, exc_value, traceback_obj)
            else:
                log_exception(exc_type, exc_value, traceback_obj)
        finally:
            self.close()

        # Reraise exception higher up the stack
        return False


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This will not filter log records, so make sure only to send
    messages that you actually want in the log to it.
    """

    timeout = 1

    def handle(self):
        """Handle requests."""
        while True:
            try:
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack(">L", chunk)[0]
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk = chunk + self.connection.recv(slen - len(chunk))
                obj = pickle.loads(chunk)
                record = logging.makeLogRecord(obj)
                self.handle_log_record(record)
            except TimeoutError:
                if self.server.is_shutting_down.is_set():
                    break

    def handle_log_record(self, record):
        """Handle a log record."""
        logger = logging.getLogger(LogRecordSocketReceiver.LISTENER_LOGGER)
        logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """A simple TCP server that will listen for log records from on a TCP-socket.

    The caller is responsible for making sure that the server is properly shutdown
    by calling:

    ```
    server.shutdown()
    server.server_close()
    ```

    Please note that this must be called while `server_forever` is running from
    a separate thread, otherwise it will deadlock.

    """

    LISTENER_LOGGER = "pixelator-logger-listener"

    allow_reuse_address = True

    def __init__(
        self,
        host=LOCALHOST,
        port=DEFAULT_TCP_LOGGING_PORT,
        handler=LogRecordStreamHandler,
        log_file=None,
        console_log_formatter=DefaultCliFormatter(),
    ):
        """Initialize the log record socket receiver."""
        self.timeout = 0.1
        self.log_file = log_file
        self._console_log_formatter = console_log_formatter
        # Use this threading.Event to make sure that the handler
        # shuts down properly when the server is closed.
        self.is_shutting_down = threading.Event()
        self._configure_logger()
        super().__init__((host, port), handler)

    def _configure_logger(self):
        """Configure the listener logger.

        This will add a file logger if a log file has been specified,
        and it will always output to the console (with the provided
        formatter).
        """
        logger = logging.getLogger(self.LISTENER_LOGGER)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Reset the handlers to make sure they are not duplicated
        # if multiple server instances are created.
        logger.handlers = []

        handlers = []
        if self.log_file:
            handler = logging.FileHandler(str(self.log_file), mode="w")
            formatter = logging.Formatter(
                "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
            )
            handler.setFormatter(formatter)
            handlers.append(handler)

        console_handler = ClickHandler()
        console_handler.setFormatter(self._console_log_formatter)
        handlers.append(console_handler)

        logger.handlers = handlers

    def server_close(self) -> None:
        """Close the server."""
        # Ensure any outstanding requests are handled before shutting down
        self.is_shutting_down.set()
        self.handle_request()

        # TODO If we don't sleep here the tests get very flakey
        # And I haven't been able to figure out any other way to fix it.
        time.sleep(0.01)

        logger = logging.getLogger(self.LISTENER_LOGGER)
        for handler in logger.handlers:
            handler.close()
