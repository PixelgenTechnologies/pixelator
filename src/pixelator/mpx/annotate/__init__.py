"""Functions for filtering and annotating of pixel data in edge list format.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
import typing
from pathlib import Path
from typing import Literal, Optional

import numba
import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from graspologic_native import leiden

from pixelator import __version__
from pixelator.common.annotate import filter_components_sizes
from pixelator.common.annotate.aggregates import call_aggregates
from pixelator.common.annotate.cell_calling import find_component_size_limits
from pixelator.common.annotate.constants import (
    MINIMUM_NBR_OF_CELLS_FOR_ANNOTATION,
)
from pixelator.common.config import AntibodyPanel
from pixelator.common.exceptions import PixelatorBaseException
from pixelator.common.report.models import SummaryStatistics
from pixelator.mpx.graph.utils import components_metrics, edgelist_metrics
from pixelator.mpx.pixeldataset import SIZE_DEFINITION, PixelDataset
from pixelator.mpx.pixeldataset.utils import edgelist_to_anndata
from pixelator.mpx.report.models.annotate import AnnotateSampleReport

# TODO
# Work around for issue with numba and multithreading
# I got the solution from here:
# https://stackoverflow.com/questions/68131348/training-a-python-umap-model-hangs-in-a-multiprocessing-process
numba.config.THREADING_LAYER = "omp"

logger = logging.getLogger(__name__)


class NoCellsFoundException(PixelatorBaseException):
    """Raised when no cells are found in the edge list."""

    pass


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

    # Calculate some metrics on the components that fail the size filter
    input_cell_count = component_metrics.shape[0]
    input_read_count = component_metrics.loc[:, "reads"].sum()
    size_filter_fail_mask = ~is_filtered_arr
    size_filter_fail_cell_count = np.sum(size_filter_fail_mask, dtype=int)
    size_filter_fail_molecule_count = component_metrics.loc[
        size_filter_fail_mask, "molecules"
    ].sum()
    size_filter_fail_read_count = component_metrics.loc[
        size_filter_fail_mask, "reads"
    ].sum()

    # save the components metrics (raw)
    component_info_file_path = (
        Path(output) / f"{output_prefix}.raw_components_metrics.csv.gz"
    )
    component_metrics.to_csv(
        component_info_file_path,
        header=True,
        index=True,
        sep=",",
        compression="gzip",
    )

    # filter the components metrics using the is_filtered flag
    filtered_component_metrics = component_metrics[is_filtered_arr]

    # filter the edge list
    filtered_edgelist = edgelist[
        edgelist["component"].isin(filtered_component_metrics.index)
    ]

    # convert the filtered edge list to AnnData
    adata = edgelist_to_anndata(edgelist=filtered_edgelist, panel=panel)

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
    dataset = PixelDataset.from_data(
        edgelist=filtered_edgelist, adata=adata, metadata=metadata
    )
    dataset.save(
        Path(output) / f"{output_prefix}.annotate.dataset.pxl", force_overwrite=True
    )

    edgelist_metrics_dict = edgelist_metrics(filtered_edgelist)
    adata_metrics_dict = anndata_metrics(adata)

    molecules_per_a_pixel_per_cell_stats = SummaryStatistics.from_series(
        filtered_component_metrics["mean_molecules_per_a_pixel"]
    )
    a_pixel_count_per_b_pixel_per_cell_stats = SummaryStatistics.from_series(
        filtered_component_metrics["mean_a_pixels_per_b_pixel"]
    )
    b_pixel_count_per_a_pixel_per_cell_stats = SummaryStatistics.from_series(
        filtered_component_metrics["mean_b_pixels_per_a_pixel"]
    )

    report = AnnotateSampleReport(
        sample_id=output_prefix,
        marker_count=edgelist_metrics_dict["marker_count"],
        a_pixel_count=edgelist_metrics_dict["a_pixel_count"],
        b_pixel_count=edgelist_metrics_dict["b_pixel_count"],
        molecule_count_per_a_pixel_per_cell_stats=molecules_per_a_pixel_per_cell_stats,
        a_pixel_count_per_b_pixel_per_cell_stats=a_pixel_count_per_b_pixel_per_cell_stats,
        b_pixel_count_per_a_pixel_per_cell_stats=b_pixel_count_per_a_pixel_per_cell_stats,
        fraction_molecules_in_largest_component=edgelist_metrics_dict[
            "fraction_molecules_in_largest_component"
        ],
        fraction_pixels_in_largest_component=edgelist_metrics_dict[
            "fraction_pixels_in_largest_component"
        ],
        **adata_metrics_dict,
        input_cell_count=input_cell_count,
        input_read_count=input_read_count,
        size_filter_fail_cell_count=size_filter_fail_cell_count,
        size_filter_fail_molecule_count=size_filter_fail_molecule_count,
        size_filter_fail_read_count=size_filter_fail_read_count,
    )

    # save metrics (JSON)
    report.write_json_file(Path(metrics_file), indent=4)


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


class AnnotateAnndataStatistics(typing.TypedDict):
    """Typed dict for the statistics of an AnnData object."""

    cell_count: int

    total_marker_count: int

    molecule_count: int

    read_count: int

    molecule_count_per_cell_stats: SummaryStatistics
    read_count_per_cell_stats: SummaryStatistics
    a_pixel_count_per_cell_stats: SummaryStatistics
    b_pixel_count_per_cell_stats: SummaryStatistics
    marker_count_per_cell_stats: SummaryStatistics
    a_pixel_b_pixel_ratio_per_cell_stats: SummaryStatistics

    aggregate_count: Optional[int]
    reads_in_aggregates_count: Optional[int]
    molecules_in_aggregates_count: Optional[int]

    min_size_threshold: Optional[int]
    max_size_threshold: Optional[int]
    doublet_size_threshold: Optional[int]
    fraction_potential_doublets: Optional[float]
    n_edges_to_split_potential_doublets: Optional[int]


def anndata_metrics(adata: AnnData) -> AnnotateAnndataStatistics:
    """Collect metrics from an AnnData object.

    :param adata: the AnnData object
    :returns: a dictionary of different metrics
    """
    molecule_count = adata.obs["molecules"].sum()
    read_count = adata.obs["reads"].sum()
    molecules_per_cell_stats = SummaryStatistics.from_series(adata.obs["molecules"])
    reads_per_cell_stats = SummaryStatistics.from_series(adata.obs["reads"])
    a_pixels_per_cell_stats = SummaryStatistics.from_series(adata.obs["a_pixels"])
    b_pixels_per_cell_stats = SummaryStatistics.from_series(adata.obs["b_pixels"])
    markers_per_cell_stats = SummaryStatistics.from_series(adata.obs["antibodies"])
    a_pixel_b_pixel_ratio_per_cell_stats = SummaryStatistics.from_series(
        adata.obs["a_pixel_b_pixel_ratio"]
    )

    metrics: AnnotateAnndataStatistics = {
        "cell_count": adata.n_obs,
        "total_marker_count": adata.n_vars,
        "molecule_count": molecule_count,
        "read_count": read_count,
        "molecule_count_per_cell_stats": molecules_per_cell_stats,
        "read_count_per_cell_stats": reads_per_cell_stats,
        "a_pixel_count_per_cell_stats": a_pixels_per_cell_stats,
        "b_pixel_count_per_cell_stats": b_pixels_per_cell_stats,
        "marker_count_per_cell_stats": markers_per_cell_stats,
        "a_pixel_b_pixel_ratio_per_cell_stats": a_pixel_b_pixel_ratio_per_cell_stats,
        "aggregate_count": None,
        "reads_in_aggregates_count": None,
        "molecules_in_aggregates_count": None,
        "min_size_threshold": None,
        "max_size_threshold": None,
        "doublet_size_threshold": None,
        "fraction_potential_doublets": None,
        "n_edges_to_split_potential_doublets": None,
    }

    # Tau type will only be available if it has been added in the annotate step
    if "tau_type" in adata.obs:
        aggregates_mask = adata.obs["tau_type"] != "normal"
        number_of_aggregates = np.sum(aggregates_mask)
        metrics["aggregate_count"] = number_of_aggregates
        metrics["reads_in_aggregates_count"] = adata[aggregates_mask].obs["reads"].sum()
        metrics["molecules_in_aggregates_count"] = (
            adata[aggregates_mask].obs["molecules"].sum()
        )

    if "min_size_threshold" in adata.uns:
        metrics["min_size_threshold"] = adata.uns["min_size_threshold"]

    if "max_size_threshold" in adata.uns:
        metrics["max_size_threshold"] = adata.uns["max_size_threshold"]

    if "doublet_size_threshold" in adata.uns:
        metrics["doublet_size_threshold"] = adata.uns["doublet_size_threshold"]

    if "is_potential_doublet" in adata.obs:
        metrics["fraction_potential_doublets"] = adata.obs[
            "is_potential_doublet"
        ].mean()

    if "n_edges_to_split_doublet" in adata.obs:
        metrics["n_edges_to_split_potential_doublets"] = adata.obs[
            "n_edges_to_split_doublet"
        ].sum()

    return metrics
