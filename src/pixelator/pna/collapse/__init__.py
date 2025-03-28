"""Module for collapsing a list of molecules into unique antibody links.

Copyright © 2024 Pixelgen Technologies AB.
"""

from .independent.collapser import RegionCollapser
from .paired.collapser import MoleculeCollapser

__all__ = ["MoleculeCollapser", "RegionCollapser"]
