"""Helpers for configuration plugin entrypoints.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import importlib.metadata
import logging
from importlib.metadata import EntryPoint
from typing import Generator

from pixelator.pna.config.config_class import PNAConfig

logger = logging.getLogger(__name__)


def fetch_config_plugins() -> Generator[EntryPoint, None, None]:
    """Find plugins and return them as in a generator.

    Returns:
        A generator of the loaded plugins

    Yields:
        The entrypoint object (EntryPoint)
    """
    eps = importlib.metadata.entry_points()
    group = "pixelator_pna.config_plugin"
    selected_entrypoints = eps.select(group=group)

    for entrypoint in selected_entrypoints:
        logger.debug("Detected config plugin %s", entrypoint.name)
        yield entrypoint


def load_config_plugins(config: PNAConfig) -> PNAConfig:
    """Load all config plugins."""
    new_config = config

    for entry_point in fetch_config_plugins():
        logger.debug("Loading config plugin %s", entry_point.name)
        new_config = entry_point.load()(new_config)

    return new_config
