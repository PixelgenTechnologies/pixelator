"""
This module contains functions for the filtering and annotation of
pixel data in edge list format.

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numba
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from pixelator import __version__
from pixelator.annotate.aggregates import call_aggregates
from pixelator.annotate.cell_calling import find_component_size_limits
from pixelator.annotate.constants import (
    MINIMUM_NBR_OF_CELLS_FOR_ANNOTATION,
)
from pixelator.graph.utils import components_metrics, edgelist_metrics
from pixelator.config import AntibodyPanel
from pixelator.pixeldataset import (
    SIZE_DEFINITION,
    PixelDataset,
    edgelist_to_anndata,
)
from pixelator.utils import np_encoder

# TODO
# Work around for issue with numba and multithreading
# I got the solution from here:
# https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
numba.config.THREADING_LAYER = "omp"

logger = logging.getLogger(__name__)


def filter_components_sizes(
    component_sizes: np.ndarray,
    min_size: Optional[int],
    max_size: Optional[int],
) -> np.ndarray:
    """
    Filter the component sizes provided in `component_sizes` using the size
    cut-offs defined in `min_size` and `max_size`. The components are not
    actually filtered, the function returns a boolean numpy array which
    evaluates to True if the component pass the filters.

    :param component_sizes: a numpy array with the size of each component
    :param min_size: the minimum size a component must have
    :param max_size: the maximum size a component must have
    :returns: a boolean np.array with the filtered status (True if the component
              pass the filters)
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
            raise RuntimeError("None of the components pass the filters")

        logger.debug(
            "Filtering resulted in %i components that pass the filters",
            n_components,
        )
    return filter_arr


def annotate_components(
    input: str,
    panel: AntibodyPanel,
    output: str,
    output_prefix: str,
    metrics_file: str,
    min_size: Optional[int],
    max_size: Optional[int],
    dynamic_filter: Optional[Literal["both", "min", "max"]],
    aggregate_calling: bool,
    verbose: bool,
) -> None:
    """
    This function takes as input an `edge list` dataframe (csv) that has been
    generated with `pixelator cluster`.

    The function filters the components in the edge list by component size
    (`min_size` and `max_size`) which is defined as the number of edges.
    Note that the large components (above `max_size`) are not filtered out
    but are marked as `is_filtered`.

    When `dynamic_filter` is "min" the `min_size` will be overruled by using a rank
    based approach, if it is "max" the `max_size` will be overruled, and if it is
    "both" both `min_size` and `max_size` will be overruled. This argument can be
    None to disable the dynamic filtering.

    The raw component metrics with the `is_filtered` column are stored in a csv file
    before filtering the edge list. This allows debugging and the generation of
    figures in the report.

    The filtered edge list is then converted to an `AnnData` object. The values used
    for filtering the components are stored in the `uns` layer.

    When `aggregate_calling` is True aggregates will be called based on marker
    specificity. This will add to keys to the AnnDAta object's `obs`,
    `is_aggregate` and `tau`. The former gives an estimate of if the component
    is an aggregate or not, and the later contains the computed tau specificity
    score.

    The following files and QC figures are generated after the filtering:

    - a dataframe with the components metrics before filtering (csv)
    - a PixelDataset with the filtered AnnData and edge list (zip)

    :param input: the path to the edge list dataframe (csv)
    :param panel: the AntibodyPanel of the panel used to generate the data
    :param output: the path to the output folder
    :param output_prefix: the prefix to prepend to the files (sample name)
    :param metrics_file: the path to a JSON file to write metrics
    :param min_size: the minimum size a component must have
    :param max_size: the maximum size a component must have
    :param dynamic_filter: use a rank based approach to define the min and or max size
    :param aggregate_calling: activate aggregate calling
    :param verbose: run if verbose mode when true
    :returns: None
    :raises: RuntimeError if max_size is smaller than min_size
    """
    logger.debug("Parsing edge list %s", input)

    # load data (edge list in data frame format)
    edgelist = pd.read_csv(input, dtype_backend="pyarrow")

    # get component metrics from the edge list
    component_metrics = components_metrics(edgelist=edgelist)

    # get the distribution of component sizes
    component_sizes = component_metrics[SIZE_DEFINITION].to_numpy()

    # obtain min/max size bounds using a dynamic rank based approach
    if dynamic_filter in ["min", "both"]:
        min_size = find_component_size_limits(
            component_sizes=component_sizes, direction="lower"
        )
    if dynamic_filter in ["max", "both"]:
        max_size = find_component_size_limits(
            component_sizes=component_sizes, direction="upper"
        )

    # check that mask size is not smaller than min size
    if max_size is not None and min_size is not None and max_size <= min_size:
        raise RuntimeError(
            f"max_size={max_size} must be greater than min_size={min_size}"
        )

    # filter the components by size, the function does no filter
    # the component sizes, it returns a numpy array evaluates to True
    # if the component pass the filters
    is_filtered_arr = filter_components_sizes(
        component_sizes=component_sizes,
        min_size=min_size,
        max_size=max_size,
    )
    component_metrics["is_filtered"] = is_filtered_arr

    # save the components metrics (raw)
    component_metrics.to_csv(
        Path(output) / f"{output_prefix}.raw_components_metrics.csv.gz",
        header=True,
        index=True,
        sep=",",
        compression="gzip",
    )

    # filter the components metrics using the is_filtered flag
    component_metrics = component_metrics[is_filtered_arr]

    # filter the edge list
    edgelist = edgelist[edgelist["component"].isin(component_metrics.index)]

    # convert the filtered edge list to AnnData
    adata = edgelist_to_anndata(edgelist=edgelist, panel=panel)

    # components clustering
    if adata.n_obs > MINIMUM_NBR_OF_CELLS_FOR_ANNOTATION:
        # perform the unsupervised clustering
        cluster_components(adata=adata, inplace=True)
    else:
        logger.warning(
            ("Skipping clustering since there are less than %s components"),
            MINIMUM_NBR_OF_CELLS_FOR_ANNOTATION,
        )

    if aggregate_calling:
        call_aggregates(adata, inplace=True)

    # create filtered PixelDataset and save it
    adata.uns["version"] = __version__
    metadata = {"version": __version__, "sample": output_prefix}
    dataset = PixelDataset.from_data(edgelist=edgelist, adata=adata, metadata=metadata)
    dataset.save(str(Path(output) / f"{output_prefix}.dataset.pxl"))

    # save metrics (JSON)
    with open(metrics_file, "w") as outfile:
        json.dump(edgelist_metrics(edgelist), outfile, default=np_encoder)


def cluster_components(
    adata: AnnData,
    obsmkey: Optional[Literal["denoised", "clr", "log1p", "normalized_rel"]] = "clr",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
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
    :returns: a new Anndata object if `inplace` is `True` or None
    :raises: AssertionError if `obsmkey` is missing
    """
    if obsmkey not in adata.obsm:
        raise AssertionError(f"Input AnnData is missing '{obsmkey}'")

    logger.debug("Computing clustering for %i components", adata.n_obs)

    if not inplace:
        adata = adata.copy()

    # perform the clustering (using default values)
    # TODO include the hyperparameters arguments in the functions
    sc.pp.neighbors(adata, use_rep=obsmkey, n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.5)
    sc.tl.leiden(adata, resolution=1.0)

    logger.debug("Clustering computed %i clusters", adata.obs["leiden"].nunique())
    return None if inplace else adata
