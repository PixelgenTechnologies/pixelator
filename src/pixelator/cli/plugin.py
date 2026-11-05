"""Plugin helpers for the cli.

Copyright © 2022 Pixelgen Technologies AB.
"""

import importlib.metadata
import logging
from importlib.metadata import EntryPoint
from typing import Generator

from click import Group

logger = logging.getLogger(__name__)


def fetch_cli_plugins() -> Generator[EntryPoint, None, None]:
    """Find plugins and return them as in a generator.

    Returns:
        A generator of the EntryPoint objects

    Yields:
        The entrypoint object (EntryPoint)
    """
    eps = importlib.metadata.entry_points()
    group = "pixelator.cli_plugin"
    selected_entrypoints = eps.select(group=group)

    for entrypoint in selected_entrypoints:
        logger.debug("Detected cli plugin %s", entrypoint.name)
        yield entrypoint


def add_cli_plugins(group: Group) -> None:
    """Add all cli plugins we can find to the provided group.

    Args:
        group: An instance of `click.Group` to add sub commands to
    Returns:
        None
    """
    for entrypoint in fetch_cli_plugins():
        logger.debug("Loading cli plugin %s", entrypoint.name)
        group.add_command(entrypoint.load())
