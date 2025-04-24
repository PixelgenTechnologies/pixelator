"""Module for aggregating pixeldatasets.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
from typing import (
    List,
)

import pandas as pd
import polars as pl
from anndata import AnnData
from anndata import concat as concatenate_anndata

from pixelator.mpx.pixeldataset import PixelDataset
from pixelator.mpx.pixeldataset.precomputed_layouts import aggregate_precomputed_layouts
from pixelator.mpx.pixeldataset.utils import (
    _enforce_edgelist_types,
    update_metrics_anndata,
)

logger = logging.getLogger(__name__)


def _get_attr_and_index_by_component(attribute, sample_names, datasets):
    for name, dataset in zip(sample_names, datasets):
        attr = getattr(dataset, attribute, None)
        if attr is not None:
            attr["sample"] = pd.Categorical([name] * len(attr))
            attr["component"] = pd.Categorical(
                attr["component"].astype(str) + "_" + attr["sample"].astype(str)
            )
            attr.set_index(["component"])
            yield attr


def _concatenate_edgelists(datasets, sample_names):
    with pl.StringCache():
        concatenated = datasets[0].edgelist_lazy
        concatenated = concatenated.with_columns(
            sample=pl.lit(sample_names[0], dtype=pl.Categorical)
        )
        concatenated = concatenated.collect()

        for idx, subsequent in enumerate(datasets[1:], start=1):
            concatenated = concatenated.extend(
                subsequent.edgelist_lazy.collect().with_columns(
                    sample=pl.lit(sample_names[idx], dtype=pl.Categorical)
                )
            )

        concatenated = concatenated.with_columns(
            component=pl.concat_str(
                pl.col("component"), pl.col("sample"), separator="_"
            )
        )

    return concatenated


def simple_aggregate(
    sample_names: List[str],
    datasets: List[PixelDataset],
    ignore_edgelists: bool = False,
) -> PixelDataset:
    """Aggregate samples in a simple way (see caveats).

    Aggregating samples in a simplistic fashion. This function should only
    be used if the dataset you merge have been generated with the same panel.

    It will concatenate all dataframes in the underlying PixelDataset instances,
    and add a new column called sample. New indexes will be formed from from the
    `sample` and `component` columns.

    The metadata dictionary will contain one key per sample.

    Since the edgelists will take up considerable amounts of memory it is possible to
    ignore them when aggregating the data. This will mean that they are not directly
    available for downstream analysis.

    :param sample_names: an iterable of the sample names to use for each dataset
    :param datasets: an iterable of the datasets you want to aggregate
    :param ignore_edgelists: ignoring merging the edgelists, leaving them empty in the
                             resulting PixelDataset. Defaults to False.

    :raises AssertionError: If not all pre-conditions are meet.
    :return: a PixelDataset instance with all the merged samples
    :rtype: PixelDataset
    """
    if not (len(datasets)) > 1:
        raise AssertionError(
            "There must be two or more datasets and names passed to `aggregate`"
        )
    if not len(sample_names) == len(datasets):
        raise AssertionError(
            "There must be as many sample names provided as there are dataset"
        )
    if not len(set(sample_names)) == len(sample_names):
        raise AssertionError("All provided sample names must be unique")

    all_var_identical = all(
        map(
            lambda x: x.adata.var.index.equals(datasets[0].adata.var.index),
            datasets,
        )
    )
    if not all_var_identical:
        raise AssertionError("All datasets must have identical `vars`")

    def _add_sample_name_as_obs_col(adata, name):
        adata.obs["sample"] = name
        return adata

    tmp_adatas = concatenate_anndata(
        {
            name: _add_sample_name_as_obs_col(dataset.adata, name)
            for name, dataset in zip(sample_names, datasets)
        },
        axis=0,
        index_unique="_",
    )
    adata = AnnData(tmp_adatas.X, obs=tmp_adatas.obs, var=datasets[0].adata.var)
    update_metrics_anndata(adata=adata, inplace=True)

    if ignore_edgelists:
        edgelists = pd.DataFrame()
    else:
        edgelists = _enforce_edgelist_types(
            _concatenate_edgelists(datasets, sample_names).to_pandas()
        )

    polarizations = pd.concat(
        _get_attr_and_index_by_component(
            "polarization", datasets=datasets, sample_names=sample_names
        ),
        axis=0,
    )
    colocalizations = pd.concat(
        _get_attr_and_index_by_component(
            "colocalization", datasets=datasets, sample_names=sample_names
        ),
        axis=0,
    )
    metadata = {
        "samples": {
            name: dataset.metadata for name, dataset in zip(sample_names, datasets)
        }
    }

    precomputed_layouts = aggregate_precomputed_layouts(
        [(name, dataset) for name, dataset in zip(sample_names, datasets)],
        all_markers=set(datasets[0].adata.var.index),
    )

    return PixelDataset.from_data(
        adata=adata,
        edgelist=edgelists,
        polarization=polarizations,
        colocalization=colocalizations,
        metadata=metadata,
        precomputed_layouts=precomputed_layouts,
        copy=False,
        allow_empty_edgelist=ignore_edgelists,
    )
