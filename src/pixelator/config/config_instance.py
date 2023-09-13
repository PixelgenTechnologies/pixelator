"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

from pixelator.config.config_class import (
    Config,
    load_assays_package,
    load_panels_package,
)
from pixelator.config.plugin import load_config_plugins

config = Config()

assays_package = "pixelator.resources.assays"
config = load_assays_package(config, assays_package)

panels_package = "pixelator.resources.panels"
config = load_panels_package(config, panels_package)

config = load_config_plugins(config)
