"""
Plugin helpers for the cli.

Copyright © 2022 Pixelgen Technologies AB.
"""

import importlib.metadata

try:
    from importlib.metadata import EntryPoints
except ImportError:
    # Python 3.8 and 3.9
    pass

import logging
from importlib.metadata import EntryPoint
from typing import Generator, List, Union

from click import Group

logger = logging.getLogger(__name__)


def fetch_cli_plugins() -> Generator[EntryPoint, None, None]:
    """
    Find plugins and return them as in a generator.

    :yields EntryPoint: The entrypoint object
    :returns: A generator of the EntryPoint objects
    """
    eps = importlib.metadata.entry_points()
    group = "pixelator.cli_plugin"
    selected_entrypoints: Union[List[EntryPoint], EntryPoints]

    if hasattr(eps, "select"):
        # New interface in Python 3.10 and newer versions of the
        # importlib_metadata backport.
        selected_entrypoints = eps.select(group=group)
    else:
        # Older interface, deprecated in Python 3.10 and recent
        # importlib_metadata, but we need it in Python 3.8 and 3.9.
        selected_entrypoints = eps.get(group, [])  # type: ignore

    for entrypoint in selected_entrypoints:
        logger.debug("Detected cli plugin %s", entrypoint.name)
        yield entrypoint


def add_cli_plugins(group: Group) -> None:
    """Add all cli plugins we can find to the provided group.

    :param group: An instance of `click.Group` to add sub commands to
    :returns: None
    """
    for entrypoint in fetch_cli_plugins():
        logger.debug("Loading cli plugin %s", entrypoint.name)
        group.add_command(entrypoint.load())
