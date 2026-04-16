"""Constants used in the graph step.

Copyright (c) 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

DEFAULT_WORKING_DIR = Path("/tmp")

LEIDEN_RANDOM_SEED = 1
MIN_PNA_COMPONENT_SIZE = 8000  # Smaller components are not considered cells
MAX_FRONTIER_SIZE_IN_CYCLE_SEARCH = 10000  # In cycle verification, limit the frontier size (number of visited nodes) to avoid excessive memory usage
MAX_CYCLE_SEARCH_STEPS = 13  # In cycle verification, limit the maximum number of steps in the shortest path search to avoid long runtimes
