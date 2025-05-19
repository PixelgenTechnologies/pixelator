"""Functions for filtering and annotating of pixel data in edge list format.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from typing import Literal, Optional

import numba
import numpy as np
import pandas as pd
from anndata import AnnData
from graspologic_native import leiden

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


def _cluster_components_using_leiden(
    adata: AnnData, resolution: float = 1.0, random_seed: Optional[int] = None
) -> None:
    """Carry out a leiden clustering on the components."""
    # It should be ok to run this over all vs all even on a dense matrix
    # since it shouldn't apply to more than a few thousande components.
    connections = adata.obsp["connectivities"]
    edgelist = list(
        (adata.obs.index[i], adata.obs.index[j], connections[i, j])
        for i, j in zip(*connections.nonzero())
    )
    _, partitions = leiden(
        edgelist,
        resolution=resolution,
        seed=random_seed,
        use_modularity=True,
        # These parameters are used to sync up the native implementation with
        # the python implementation we originally used.
        iterations=1,
        randomness=0.001,
        trials=1,
        starting_communities=None,
    )
    partitions_df = pd.DataFrame.from_dict(partitions, orient="index")
    adata.obs["leiden"] = partitions_df
    adata.obs = adata.obs.astype({"leiden": "category"})


def cluster_components(
    adata: AnnData,
    obsmkey: Optional[Literal["clr", "log1p"]] = "clr",
    inplace: bool = True,
    random_seed: Optional[int] = None,
) -> Optional[AnnData]:
    """Cluster the components based on their antibody counts.

    Clusters components based on their clr transformed antibody counts using
    the k-nearest neighbors, UMAP and leiden algorithms.

    It requires that the `obsmkey` is  present in the `obsm`
    layer of the input `adata` object.

    A new column called `leiden` will be added to `obs` containing the
    cluster ids.

    A new column called `X_umap` will be added to `obsm` containing the
    coordinates of the UMAP manifold.

    :param adata: AnnData object to do the clustering on
    :param obsmkey: Key to access the values `obsm` layer of `adata`
    :param inplace: If `True` performs the operation inplace on `adata`
    :param random_seed: If set this seed will be used to seed the random number
                        generators used when calculating neighbors, building the umap
                        and for the leiden clustering.
    :returns: a new Anndata object if `inplace` is `True` or None
    :rtype: Optional[AnnData]
    :raises: AssertionError if `obsmkey` is missing
    """
    # Import here as it is a slow import
    import scanpy as sc

    if obsmkey not in adata.obsm:
        raise AssertionError(f"Input AnnData is missing '{obsmkey}'")

    logger.debug("Computing clustering for %i components", adata.n_obs)

    if not inplace:
        adata = adata.copy()

    # perform the clustering (using default values)
    sc.pp.neighbors(
        adata,
        use_rep=obsmkey,
        n_neighbors=15,
        random_state=random_seed if random_seed else 0,
    )
    sc.tl.umap(adata, min_dist=0.5, random_state=random_seed if random_seed else 0)
    _cluster_components_using_leiden(
        adata, resolution=1.0, random_seed=random_seed if random_seed else None
    )

    logger.debug("Clustering computed %i clusters", adata.obs["leiden"].nunique())
    return None if inplace else adata
