"""
Console script for pixelator (common functions)

Copyright © 2022 Pixelgen Technologies AB.
"""

import collections
import functools
import logging.handlers
from pathlib import Path
from typing import Dict, Mapping, Optional

import click

BASE_DIR = str(Path(__file__).parent)
logger = logging.getLogger("pixelator.cli")


# code snipped obtained from
# https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
# the purpose is to order subcommands in order of addition
class OrderedGroup(click.Group):
    """Custom click.Group that keeps insertion order for subcommands."""

    def __init__(  # noqa: D107
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
        """Return a list of subcommands."""
        return self.commands


def output_option(func):
    """Wrap a Click entrypoint to add the --output option."""

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
    """Decorate a click command and add the --design option."""
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
