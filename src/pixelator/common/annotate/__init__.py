"""Functions for filtering and annotating of pixel data in edge list format.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from typing import Optional

import numba
import numpy as np

from pixelator.common.exceptions import PixelatorBaseException

# Work around for issue with numba and multithreading
# I got the solution from here:
# https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
numba.config.THREADING_LAYER = "omp"

logger = logging.getLogger(__name__)


class NoCellsFoundException(PixelatorBaseException):
    """Raised when no cells are found in the edge list."""

    pass


def filter_components_sizes(
    component_sizes: np.ndarray,
    min_size: Optional[int],
    max_size: Optional[int],
) -> np.ndarray:
    """Filter components by size.

    Filter the component sizes provided in `component_sizes` using the size
    cut-offs defined in `min_size` and `max_size`. The components are not
    actually filtered, the function returns a boolean numpy array which
    evaluates to True if the component pass the filters.

    :param component_sizes: a numpy array with the size of each component
    :param min_size: the minimum size a component must have
    :param max_size: the maximum size a component must have
    :returns: a boolean np.array with the filtered status (True if the component
              pass the filters)
    :rtype: np.ndarray
    :raises: RuntimeError if all the components are filtered
    """
    n_components = len(component_sizes)
    logger.debug(
        "Filtering %i components using min-size=%s and max-size=%s",
        n_components,
        min_size,
        max_size,
    )

    # create a numpy array with True as default
    filter_arr = np.full((n_components), True)

    # check if any filter has been provided to the function
    if min_size is None and max_size is None:
        logger.warning("No filtering criteria provided to filter components")
    else:
        # get the components to filter (boolean array)
        if min_size is not None:
            filter_arr &= component_sizes > min_size
        if max_size is not None:
            filter_arr &= component_sizes < max_size

        # check if none of the components pass the filters
        n_components = filter_arr.sum()
        if n_components == 0:
            raise NoCellsFoundException(
                "All cells were filtered by the size filters. Consider either setting different size filters or disabling them."
            )

        logger.debug(
            "Filtering resulted in %i components that pass the filters",
            n_components,
        )
    return filter_arr
