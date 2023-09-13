"""
Console script for pixelator (common functions)

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import traceback
import collections
import functools
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Mapping, Optional
import sys
import click
import warnings

from numba.core.errors import NumbaDeprecationWarning


BASE_DIR = str(Path(__file__).parent)

pixelator_root_logger = logging.getLogger("pixelator")
logger = logging.getLogger("pixelator.cli")


# Silence deprecation warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="geopandas not available. Some functionality will be disabled.",
    category=UserWarning,
)


def init_logger(log_file: str, verbose: bool) -> None:
    """
    Helper function to create and initialize a logging object
    with the arguments given

    :param log_file: the path to the log file
    :param verbose: True to enable verbose mode (DEBUG)
    :returns: None
    """
    mode = "a" if os.path.isfile(log_file) else "w"
    handler = logging.handlers.WatchedFileHandler(log_file, mode=mode)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    pixelator_root_logger.addHandler(handler)
    pixelator_root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Disable matplot lib debug logs (that will clog all debug logs)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.ticker").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.ERROR)

    def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Will call default excepthook
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        message = traceback.print_exception(exc_type, exc_value, exc_traceback)
        click.echo(message)

        # Create a critical level log message with info from the except hook.
        pixelator_root_logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Assign the excepthook to the handler
    sys.excepthook = handle_unhandled_exception


# code snipped obtained from
# https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
# the purpose is to order subcommands in order of addition
class OrderedGroup(click.Group):
    def __init__(
        self,
        name: Optional[str] = None,
        commands: Optional[Dict[str, click.Command]] = None,
        **kwargs,
    ):
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        self.commands = commands or collections.OrderedDict()

    def list_commands(  # type: ignore
        self, ctx: click.Context
    ) -> Mapping[str, click.Command]:
        return self.commands


def output_option(func):
    @click.option(
        "--output",
        required=True,
        type=click.Path(exists=False),
        help=(
            "The path where the results will be placed (it is created if it does not"
            " exist)"
        ),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def design_option(func):
    from pixelator.config import config

    assay_options = list(config.assays.keys())

    @click.option(
        "--design",
        required=True,
        default=None,
        type=click.Choice(assay_options),
        help="The design to load from the configuration file",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
