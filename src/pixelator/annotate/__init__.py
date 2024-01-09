"""Functions for filtering and annotating of pixel data in edge list format.

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import json
import logging
import typing
from pathlib import Path
from typing import Literal, Optional

import numba
import numpy as np
import polars as pl
import scanpy as sc
from anndata import AnnData

from pixelator import __version__
from pixelator.annotate.aggregates import call_aggregates
from pixelator.annotate.cell_calling import find_component_size_limits
from pixelator.annotate.constants import (
    MINIMUM_NBR_OF_CELLS_FOR_ANNOTATION,
)
from pixelator.config import AntibodyPanel
from pixelator.graph.utils import components_metrics, edgelist_metrics
from pixelator.pixeldataset import (
    SIZE_DEFINITION,
    PixelDataset,
)
from pixelator.pixeldataset.utils import edgelist_to_anndata
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
    """Create a pixeldataset from a provided edgelist.

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
    is an aggregate or not, and the latter contains the computed tau specificity
    score.

    The following files and QC figures are generated after the filtering:

    - a dataframe with the components metrics before filtering (csv)
    - a PixelDataset with the filtered AnnData and edge list (zip)

    :param input: the path to the edge list dataframe (parquet)
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
    :rtype: None
    :raises: RuntimeError if max_size is smaller than min_size
    """
    logger.debug("Parsing edge list %s", input)

    # load data (edge list in data frame format)
    edgelist = pl.read_parquet(input).to_pandas()

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

    edgelist_metrics_dict = edgelist_metrics(edgelist)
    adata_metrics_dict = anndata_metrics(adata)
    annotate_metrics = {**edgelist_metrics_dict, **adata_metrics_dict}

    # save metrics (JSON)
    with open(metrics_file, "w") as outfile:
        json.dump(annotate_metrics, outfile, default=np_encoder)


def cluster_components(
    adata: AnnData,
    obsmkey: Optional[Literal["denoised", "clr", "log1p", "normalized_rel"]] = "clr",
    inplace: bool = True,
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
    :returns: a new Anndata object if `inplace` is `True` or None
    :rtype: Optional[AnnData]
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


class AnnotateAnndataStatistics(typing.TypedDict):
    cells_filtered: int
    total_markers: int
    total_umis: int
    total_reads_cell: int
    median_reads_cell: int
    mean_reads_cell: float
    median_upi_cell: int
    mean_upi_cell: float
    median_upia_cell: int
    mean_upia_cell: float
    median_umi_cell: int
    mean_umi_cell: float
    median_umi_upia_cell: float
    mean_umi_upia_cell: float
    median_upia_degree_cell: float
    mean_upia_degree_cell: float
    median_markers_cell: int
    mean_markers_cell: float
    upib_per_upia: float

    # Aggregate calling metrics
    number_of_aggregates: Optional[int]
    fraction_of_aggregates: Optional[float]
    reads_of_aggregates: Optional[float]
    umis_of_aggregates: Optional[float]

    min_size_threshold: Optional[int]
    max_size_threshold: Optional[int]
    doublet_size_threshold: Optional[int]


def anndata_metrics(adata: AnnData) -> dict[str, float]:
    """Collect metrics from an AnnData object.

    A helper private function to collect metrics from an AnnData object.

    :param adata: the AnnData object
    :returns: a dictionary of different metrics
    :rtype: Dict[str, float]
    """
    metrics: AnnotateAnndataStatistics = {
        "cells_filtered": adata.n_obs,
        "total_markers": adata.n_vars,
        "total_umis": adata.obs["umi"].sum(),
        "total_reads_cell": adata.obs["reads"].sum(),
        "median_reads_cell": adata.obs["reads"].median(),
        "mean_reads_cell": adata.obs["reads"].mean(),
        "median_upi_cell": adata.obs["vertices"].median(),
        "mean_upi_cell": adata.obs["vertices"].mean(),
        "median_upia_cell": adata.obs["upia"].median(),
        "mean_upia_cell": adata.obs["upia"].mean(),
        "median_umi_cell": adata.obs["umi"].median(),
        "mean_umi_cell": adata.obs["umi"].mean(),
        "median_umi_upia_cell": adata.obs["median_umi_per_upia"].median(),
        "mean_umi_upia_cell": adata.obs["mean_umi_per_upia"].mean(),
        "median_upia_degree_cell": adata.obs["median_upia_degree"].median(),
        "mean_upia_degree_cell": adata.obs["mean_upia_degree"].mean(),
        "median_markers_cell": adata.obs["antibodies"].median(),
        "mean_markers_cell": adata.obs["antibodies"].mean(),
        "upib_per_upia": adata.obs["upib"].sum() / adata.obs["upia"].sum(),
        "number_of_aggregates": None,
        "fraction_of_aggregates": None,
        "reads_of_aggregates": None,
        "umis_of_aggregates": None,
        "min_size_threshold": None,
        "max_size_threshold": None,
        "doublet_size_threshold": None,
    }

    # This metric needs to be initialized for the webreport
    metrics["reads_of_aggregates"] = 0
    metrics["umis_of_aggregates"] = 0

    # Tau type will only be available if it has been added in the annotate step
    if "tau_type" in adata.obs:
        aggregates_mask = adata.obs["tau_type"] != "normal"
        number_of_aggregates = np.sum(aggregates_mask)
        metrics["number_of_aggregates"] = number_of_aggregates
        metrics["fraction_of_aggregates"] = number_of_aggregates / len(
            adata.obs["tau_type"]
        )
        metrics["reads_of_aggregates"] = adata[aggregates_mask].obs["reads"].sum()
        metrics["umis_of_aggregates"] = adata[aggregates_mask].obs["umi"].sum()

    if "min_size_threshold" in adata.uns:
        metrics["minimum_size_threshold"] = adata.uns["min_size_threshold"]

    if "max_size_threshold" in adata.uns:
        metrics["max_size_threshold"] = adata.uns["max_size_threshold"]

    if "doublet_size_threshold" in adata.uns:
        metrics["doublet_size_threshold"] = adata.uns["doublet_size_threshold"]

    return metrics
