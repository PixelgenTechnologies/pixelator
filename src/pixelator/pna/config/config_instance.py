"""Copyright © 2023 Pixelgen Technologies AB."""

from pixelator.pna.config.config_class import (
    PNAConfig,
    load_assays_package,
    load_panels_package,
)
from pixelator.pna.config.plugin import load_config_plugins

pna_config = PNAConfig()

assays_package = "pixelator.pna.resources.assays"
pna_config = load_assays_package(pna_config, assays_package)

panels_package = "pixelator.pna.resources.panels"
pna_config = load_panels_package(pna_config, panels_package)

pna_config = load_config_plugins(pna_config)
