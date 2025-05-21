"""Copyright © 2023 Pixelgen Technologies AB."""

from pixelator.common.config.config_class import (
    Config,
    load_assays_package,
    load_panels_package,
)
from pixelator.mpx.config.plugin import load_config_plugins

config = Config()

assays_package = "pixelator.mpx.resources.assays"
config = load_assays_package(config, assays_package)

panels_package = "pixelator.mpx.resources.panels"
config = load_panels_package(config, panels_package)

config = load_config_plugins(config)
