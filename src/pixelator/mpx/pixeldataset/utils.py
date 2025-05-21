"""Module for utilities for working with pixeldatasets.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from anndata import AnnData, ImplicitModificationWarning, read_h5ad
from graspologic_native import leiden

from pixelator.common.statistics import (
    clr_transformation,
    log1p_transformation,
)
from pixelator.common.types import PathType
from pixelator.mpx.graph import components_metrics
from pixelator.mpx.graph.constants import (
    LEIDEN_RESOLUTION,
    RELATIVE_ANNOTATE_RESOLUTION,
)

if TYPE_CHECKING:
    from pixelator.common.config import AntibodyPanel

logger = logging.getLogger(__name__)


def update_metrics_anndata(adata: AnnData, inplace: bool = True) -> Optional[AnnData]:
    """Update any metrics in the AnnData instance.

    This will  update the QC metrics (`var` and `obs`) of
    the AnnData object given as input. This function is typically used
    when the AnnData object has been filtered and one wants the QC metrics
    to be updated accordingly.

    :param adata: an AnnData object
    :param inplace: If `True` performs the operation inplace
    :returns: the updated AnnData object or None if inplace is True
    :rtype: Optional[AnnData]
    """
    logger.debug(
        "Updating metrics in AnnData object with %i components and %i markers",
        adata.n_obs,
        adata.n_vars,
    )

    if not inplace:
        adata = adata.copy()

    df = adata.to_df()

    # update the var layer (antibody metrics)
    # we ignore the warning here, since we actually want to force and update of the
    # `adata.var` data frame.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
        adata.var["antibody_count"] = df.sum().astype(int)
    adata.var["components"] = (df != 0).sum()
    adata.var["antibody_pct"] = (
        adata.var["antibody_count"] / adata.var["antibody_count"].sum()
    )

    # update the obs layer (components metrics)
    adata.obs["antibodies"] = np.sum(adata.X > 0, axis=1)

    logger.debug("Metrics in AnnData object updated")
    return None if inplace else adata


def _enforce_edgelist_types(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Enforce the data types of the edgelist."""
    # Enforcing the types of the edgelist reduces the memory
    # usage by roughly 2/3s.

    required_types = {
        "count": "uint16",
        "upia": "category",
        "upib": "category",
        "umi": "category",
        "marker": "category",
        "sequence": "category",
        "component": "category",
    }

    # if the dataframe is empty just enforce the types.
    if edgelist.shape[0] == 0:
        edgelist = pd.DataFrame(columns=required_types.keys())

    # If we have the optional sample column, this should be
    # set to use a categorical type
    if "sample" in edgelist.columns:
        required_types["sample"] = "category"

    # If all of the prescribed types are already set, just return the edgelist
    type_dict = edgelist.dtypes.to_dict()
    if all(type_dict[key] == type_ for key, type_ in required_types.items()):
        return edgelist

    return edgelist.astype(
        required_types,
        # Do not copy here, since otherwise the memory usage
        # blows up
        copy=False,
    )


def antibody_metrics(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for each antibody/marker.

    A helper function that computes a dataframe of antibody
    metrics for each antibody (marker) present in the edge list
    given as input. The metrics include: total count, relative
    count and the number of components where the antibody is detected.

    :param edgelist: an edge list dataframe with a membership column
    :returns: a pd.DataFrame with the antibody metrics per antibody
    :rtype: pd.DataFrame
    :raises: AssertionError when the input edge list is not valid
    """
    if "component" not in edgelist.columns:
        raise AssertionError("Edge list is missing the component column")

    logger.debug(
        "Computing antibody metrics for dataset with %i elements", edgelist.shape[0]
    )

    # compute metrics
    antibody_metrics = (
        edgelist.groupby("marker", observed=True)
        .agg(
            {
                "count": "sum",
                "component": "nunique",
            }
        )
        .astype(int)
    )
    antibody_metrics.columns = [  # type: ignore
        "antibody_count",
        "components",
    ]

    # add relative counts
    antibody_metrics["antibody_pct"] = (
        antibody_metrics["antibody_count"] / antibody_metrics["antibody_count"].sum()
    ).astype(float)

    logger.debug("Antibody metrics computed")
    return antibody_metrics


def component_antibody_counts(edgelist: pd.DataFrame) -> pd.DataFrame:
    """Calculate antibody counts per component.

    A helper function that computes a dataframe of antibody
    counts for each component present in the edge list given
    as input (component column).

    :param edgelist: an edge list dataframe with a membership column
    :returns: a pd.DataFrame with the antibody counts per component
    :rtype: pd.DataFrame
    :raises: AssertionError when the input edge list is not valid
    """
    if "component" not in edgelist.columns:
        raise AssertionError("Edge list is missing the component column")

    logger.debug(
        "Computing antibody counts for edge list with %i edges and %i markers",
        edgelist.shape[0],
        edgelist.shape[1],
    )

    # iterate the components to obtain the metrics of each component
    # TODO this seems to be memory demanding so a simpler groupby() over
    # the component column may perform better in terms of memory
    df = (
        edgelist.groupby(["component", "marker"], observed=True)
        .agg("size")
        .unstack()
        .fillna(0)
    ).astype(int)
    df.index.name = "component"

    logger.debug("Antibody counts computed")
    return df


def read_anndata(filename: str) -> AnnData:
    """Read an AnnData object from a h5ad file.

    A simple wrapper to read/parse AnnData (h5ad) files.

    :param filename: the path to the AnnData file (h5ad)
    :returns: an AnnData object
    :rtype: AnnData
    :raises: AssertionError when the input is not valid
    """
    if not os.path.isfile(filename):
        raise AssertionError(f"input {filename} does not exist")
    if not filename.endswith("h5ad"):
        raise AssertionError(f"input {filename} has a wrong extension")
    return read_h5ad(filename=filename)


def write_anndata(adata: AnnData, filename: PathType) -> None:
    """Write anndata instance to file.

    A simple wrapper to write/save an AnnData object to a file.

    :param adata: the AnnData object to be saved
    :param filename: the path to save AnnData file (h5ad)
    :returns: None
    :rtype: None
    """
    adata.write(filename=filename, compression="gzip")


def _compute_sub_communities(
    component_edgelist: pd.DataFrame, n_edges_reconnect: int | None = None
) -> pd.Series:
    component_edgelist = (
        component_edgelist.groupby(["upia", "upib"], observed=True)["count"]
        .count()
        .reset_index()
        .sort_values(["upia", "upib"])
    )
    edgelist_tuple = list(
        map(tuple, np.array(component_edgelist[["upia", "upib", "count"]]))
    )
    _, component_communities_dict = leiden(
        edgelist_tuple,
        resolution=RELATIVE_ANNOTATE_RESOLUTION * LEIDEN_RESOLUTION,
        seed=42,
        # These parameters are used to sync up the native implementation with
        # the python implementation we originally used.
        use_modularity=True,
        iterations=1,
        randomness=0.001,
        trials=1,
        starting_communities=None,
    )
    component_communities = pd.Series(component_communities_dict)

    return component_communities


def _assess_doublet(component_edgelist: pd.DataFrame) -> tuple[bool, int]:
    """Check whether a component is a potential doublet and how many edges should be removed to split it.

    A component is a potential doublet if a) it has more than one community and
    b) the second largest community is at least 20% of the size of the largest
    community. A lower resolution is to be used for annotation of potential doublets
    compared to the component recovery in the graph phase. The reduction factor in
    annotate resolution is set by RELATIVE_ANNOTATE_RESOLUTION (default is 0.5).

    """
    component_communities = _compute_sub_communities(component_edgelist)
    component_community_sizes = component_communities.value_counts().sort_values(
        ascending=False
    )
    if len(component_community_sizes) > 1 and component_community_sizes.iloc[1] > (
        0.2 * component_community_sizes.iloc[0]
    ):
        edges_to_remove = (
            component_edgelist["upia"].map(component_communities)
            != component_edgelist["upib"].map(component_communities)
        ).sum()
        return True, edges_to_remove
    else:
        return False, 0


def mark_potential_doublets(
    edgelist: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Mark whether a component is a potential doublet.

    A component is a potential doublet if a) it has more than one community and
    b) the second largest community is at least 20% of the size of the largest
    community.

    :param edgelist: the edge list dataframe containing component labels.
    :returns: a boolean series indicating whether a component is a potential doublet.
    :rtype: pd.Series
    """
    is_potential_doublet = pd.Series(index=edgelist["component"].unique(), dtype=bool)
    n_edges_to_split_doublet = pd.Series(
        index=edgelist["component"].unique(), dtype=int
    )
    for component_id, component_edgelist in edgelist.groupby("component"):
        is_potential_doublet[component_id], n_edges_to_split_doublet[component_id] = (
            _assess_doublet(component_edgelist)
        )

    return is_potential_doublet, n_edges_to_split_doublet


def edgelist_to_anndata(
    edgelist: pd.DataFrame,
    panel: AntibodyPanel,
) -> AnnData:
    """Convert an edgelist to an anndata object.

    A helper function to build an AnnData object from an edge list (dataframe).
    The `panel` will be used to add extra information (`var` layer) and to ensure
    that all the antibodies are included in the AnnData object.

    The AnnData will have the following layers:

    .X = the component to antibody counts
    .var = the antibody metrics
    .obs = the component metrics
    .obsm["clr"] = the transformed (clr) component to antibody counts
    .obsm["log1p"] = the transformed (log1p) component to antibody counts

    :param edgelist: an edge list (pd.DataFrame)
    :param panel: the AntibodyPanel of the panel used to generate the data
    :returns: an AnnData object
    :rtype: AnnData
    """
    logger.debug("Creating AnnData from edge list with %i edges", edgelist.shape[0])

    missing_markers = ",".join(
        set(panel.markers).difference(set(edgelist["marker"].unique()))
    )
    if missing_markers:
        msg = (
            "The given 'panel' is missing markers "
            f"({missing_markers}) in the edge list, "
            "these will be added with 0 counts"
        )
        logger.warning(msg)

    # compute antibody counts and re-index
    counts_df = component_antibody_counts(edgelist=edgelist)
    counts_df = counts_df.reindex(columns=panel.markers, fill_value=0)
    counts_df.index = counts_df.index.astype(str)
    counts_df.columns = counts_df.columns.astype(str)

    # compute components metrics (obs) and re-index
    components_metrics_df = components_metrics(edgelist=edgelist)
    components_metrics_df = components_metrics_df.reindex(index=counts_df.index)
    (
        components_metrics_df["is_potential_doublet"],
        components_metrics_df["n_edges_to_split_doublet"],
    ) = mark_potential_doublets(edgelist=edgelist)
    # compute antibody metrics (var) and re-index
    antibody_metrics_df = antibody_metrics(edgelist=edgelist)
    antibody_metrics_df = antibody_metrics_df.reindex(index=panel.markers, fill_value=0)
    # Do a dtype conversion of the columns here since AnnData cannot handle
    # a pyarrow arrays.
    antibody_metrics_df = antibody_metrics_df.astype(
        {"antibody_count": "int64", "antibody_pct": "float32"}
    )

    # create AnnData object
    adata = AnnData(
        X=counts_df,
        obs=components_metrics_df,
        var=antibody_metrics_df,
    )

    # add extra panel variables to var
    adata.var["nuclear"] = panel.df["nuclear"].to_numpy()
    adata.var["control"] = panel.df["control"].to_numpy()

    # add normalization layers
    counts_df_clr = clr_transformation(df=counts_df, axis=1)
    counts_df_log1p = log1p_transformation(df=counts_df)
    adata.obsm["clr"] = counts_df_clr
    adata.obsm["log1p"] = counts_df_log1p

    logger.debug("AnnData created")
    return adata
