"""
Helpers for configuration plugin entrypoints

Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations
import logging
from typing import Generator, List, TYPE_CHECKING, Union

import importlib.metadata
from importlib.metadata import EntryPoint

try:
    from importlib.metadata import EntryPoints
except ImportError:
    # Python 3.8 and 3.9
    pass

if TYPE_CHECKING:
    from pixelator.config import Config

logger = logging.getLogger(__name__)


def fetch_config_plugins() -> Generator[importlib.metadata.EntryPoint, None, None]:
    """
    Find plugins and return them as in a generator

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
        selected_entrypoints = eps.get(group, [])

    for entrypoint in selected_entrypoints:
        logger.debug("Detected config plugin %s", entrypoint.name)
        yield entrypoint


def load_config_plugins(config: Config) -> Config:
    new_config = config

    for entry_point in fetch_config_plugins():
        logger.debug("Loading config plugin %s", entry_point.name)
        new_config = entry_point.load()(new_config)

    return new_config
