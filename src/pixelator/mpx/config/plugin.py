"""Helpers for configuration plugin entrypoints.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import importlib.metadata
import logging
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Generator, List, Union

try:
    from importlib.metadata import EntryPoints
except ImportError:
    # Python 3.8 and 3.9
    pass

if TYPE_CHECKING:
    from pixelator.common.config import Config

logger = logging.getLogger(__name__)


def fetch_config_plugins() -> Generator[EntryPoint, None, None]:
    """Find plugins and return them as in a generator.

    :yields EntryPoint: The entrypoint object
    :returns: A generator of the loaded plugins
    """
    eps = importlib.metadata.entry_points()
    group = "pixelator.config_plugin"
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
        logger.debug("Detected config plugin %s", entrypoint.name)
        yield entrypoint


def load_config_plugins(config: Config) -> Config:
    """Load all config plugins."""
    new_config = config

    for entry_point in fetch_config_plugins():
        logger.debug("Loading config plugin %s", entry_point.name)
        new_config = entry_point.load()(new_config)

    return new_config
