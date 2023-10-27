"""Module for PixelDataset and associated functions.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
)

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.graph import Graph
from pixelator.pixeldataset.utils import (
    _enforce_edgelist_types,
    update_metrics_anndata,
)
from pixelator.pixeldataset.backends import (
    PixelDatasetBackend,
    FileBasedPixelDatasetBackend,
    ObjectBasedPixelDatasetBackend,
)
from pixelator.pixeldataset.file_formats import (
    PixelFileFormatSpec,
    PixelFileParquetFormatSpec,
    PixelFileCSVFormatSpec,
)
from pixelator.types import PathType

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module="libpysal")


logger = logging.getLogger(__name__)

SIZE_DEFINITION = "edges"
# minimum number of vertices (nodes) required for polarization/co-localization
MIN_VERTICES_REQUIRED = 100


def read(path: PathType) -> PixelDataset:
    """Read a PixelDataset from a provided .pxl file.

    :param path: path to the file to read
    :return: an instance of `PixelDataset`
    :rtype: PixelDataset
    """
    return PixelDataset.from_file(path)


class PixelDataset:
    """PixelDataset represents data from one or more Molecular Pixelation experiments.

    In general the way to instantiate a new PixelDataset is to use it's `from_file`
    method:

    ```
    pxl_data = PixelDataset.from_file("/path/to/file.pxl")
    ```

    adata: the AnnData object containing marker counts from the various
           components -> required
    edgelist: the edge list (pd.DataFrame). This can be used to extract spatial
              graph information for each component -> required
    metadata: a dictionary including meta information -> Optional
    polarization: the polarization scores (pd.DataFrame) for
                  each component -> Optional
    colocalization: the colocalization scores (pd.DataFrame) for each
                    component -> Optional
    """

    def __init__(self, backend: PixelDatasetBackend) -> None:
        """Create a `PixelDataset` from a `PixelDatasetBackend`.

        In general you should probably use `PixelDataset.from_file` to
        create a new `PixelDataset` instance.

        This method is reserved for advanced use.
        :param backend: an instance of `PixelDatasetBackend`
        """
        self._backend = backend

    @staticmethod
    def from_file(path: PathType) -> PixelDataset:
        """Create a new instance of `PixelDataset` from the file at the provided path.

        :param path: path to a .pxl file
        :return: A new instance of `PixelDataset`
        :rtype: PixelDataset
        """
        # We can ignore the error here, since while MyPy thinks that the
        # @cached_property decorators that we use on FileBasedPixelDatasetBackend
        # do not support setting, they actually do. See the documentation here:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        return PixelDataset(backend=FileBasedPixelDatasetBackend(path))  # type: ignore

    @staticmethod
    def from_data(
        adata: AnnData,
        edgelist: Optional[pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
        polarization: Optional[pd.DataFrame] = None,
        colocalization: Optional[pd.DataFrame] = None,
        copy: bool = True,
        allow_empty_edgelist: bool = False,
    ) -> PixelDataset:
        """Create a new instance of PixelDataset from the provided underlying objects.

        :param adata: an instance of `AnnData`
        :param edgelist: an edgelist as a `pd.DataFrame`
        :param metadata: an instance of a dictionary with metadata, defaults to None
        :param polarization: a `pd.DataFrame` with polarization information,
                             defaults to None
        :param colocalization: a `pd.DataFrame` with colocalization information,
                               defaults to None
        :param copy: specify if the input data should be copied or not.
                     Defaults to True.
        :param allow_empty_edgelist: allow the edgelist to be empty. Defaults to False.
        :return: An instance of PixelDataset
        :rtype: PixelDataset
        """
        return PixelDataset(
            backend=ObjectBasedPixelDatasetBackend(
                adata=adata,
                edgelist=edgelist,
                metadata=metadata,
                polarization=polarization,
                colocalization=colocalization,
                copy=copy,
                allow_edgelist_to_be_empty=allow_empty_edgelist,
            )
        )

    @property
    def adata(self) -> AnnData:
        """Get the AnnData object."""
        return self._backend.adata

    @adata.setter
    def adata(self, value: AnnData) -> None:
        """Set the AnnData object."""
        self._backend.adata = value

    @property
    def edgelist(self) -> pd.DataFrame:
        """Get the edge list."""
        return _enforce_edgelist_types(self._backend.edgelist)

    @edgelist.setter
    def edgelist(self, value: pd.DataFrame) -> None:
        """Set the edge list."""
        self._backend.edgelist = _enforce_edgelist_types(value)

    @property
    def edgelist_lazy(self) -> pl.LazyFrame:
        """Get the edge list as a lazy dataframe."""
        lz_edgelist = self._backend.edgelist_lazy
        if "index" in lz_edgelist.columns:
            warnings.warn(
                "A column called `index` was identified in your edgelist. "
                "This will be removed."
            )
            lz_edgelist = lz_edgelist.drop("index")
        return lz_edgelist

    @property
    def polarization(self) -> Optional[pd.DataFrame]:
        """Get the polarization object."""
        return self._backend.polarization

    @polarization.setter
    def polarization(self, value: pd.DataFrame) -> None:
        """Set the polarization object."""
        self._backend.polarization = value

    @property
    def colocalization(self) -> Optional[pd.DataFrame]:
        """Get the colocalization object."""
        return self._backend.colocalization

    @colocalization.setter
    def colocalization(self, value: pd.DataFrame) -> None:
        """Set the colocalization object."""
        self._backend.colocalization = value

    @property
    def metadata(self) -> Dict:
        """Get the metadata dictionary."""
        metadata = self._backend.metadata
        if not metadata:
            return {}
        return metadata

    @metadata.setter
    def metadata(self, value: Dict) -> None:
        """Set the metadata object."""
        self._backend.metadata = value

    def graph(
        self,
        component_id: Optional[str] = None,
        add_node_marker_counts: bool = True,
        simplify: bool = True,
        use_full_bipartite: bool = True,
    ) -> Graph:
        """Get the graph from the underlying edgelist.

        :param component_id: Optionally give the component id of the component
                             to only return that component.
        :param add_node_marker_counts: Add marker counts to the nodes of the graph
        :param simplify: If True, removes self-loops and multiple edges between nodes
                         from the graph
        :param use_full_bipartite: If True, the full bipartite graph will be used,
                                   otherwise it will return the A-node projection
        :return: A Graph instance
        :rtype: Graph
        :raises: KeyError if the provided `component_id` is not found in the edgelist
        """
        if component_id:
            potential_component = self.edgelist_lazy.filter(
                pl.col("component") == component_id
            )
            if potential_component.fetch(1).is_empty():
                raise KeyError(f"{component_id} not found in edgelist")
            return Graph.from_edgelist(
                potential_component,
                add_marker_counts=add_node_marker_counts,
                simplify=simplify,
                use_full_bipartite=use_full_bipartite,
            )
        return Graph.from_edgelist(
            self.edgelist_lazy,
            add_marker_counts=add_node_marker_counts,
            simplify=simplify,
            use_full_bipartite=use_full_bipartite,
        )

    def __str__(self) -> str:
        """Get a string representation of this object."""
        nbr_of_edges = self.edgelist_lazy.select(pl.count()).collect()[0, 0]
        msg = (
            f"Pixel dataset contains:\n"
            f"\tAnnData with {self.adata.n_obs} obs and {self.adata.n_vars} vars\n"
            f"\tEdge list with {nbr_of_edges} edges"
        )

        if self.polarization is not None:
            msg += f"\n\tPolarization scores with {self.polarization.shape[0]} elements"

        if self.colocalization is not None:
            msg += (
                "\n\tColocalization scores with "
                f"{self.colocalization.shape[0]} elements"
            )

        if self.metadata is not None:
            msg += "\n\tMetadata:\n"
            msg += "\n".join(
                [f"\t\t{key}: {value}" for key, value in self.metadata.items()]
            )

        msg = msg.replace("\t", "  ")

        return msg

    def __repr__(self) -> str:
        """Get a string representation of this object."""
        return str(self)

    def copy(self) -> PixelDataset:
        """Create a copy of the current PixelDataset instance.

        :return: A copy of the PixelDataset instance
        :rtype: PixelDataset
        """
        return PixelDataset.from_data(
            adata=self.adata.copy(),
            edgelist=self.edgelist.copy(),
            polarization=self.polarization.copy()
            if self.polarization is not None
            else None,
            colocalization=self.colocalization.copy()
            if self.colocalization is not None
            else None,
            metadata=self.metadata.copy() if self.metadata is not None else None,
        )

    def save(
        self,
        path: PathType,
        file_format: Literal["csv", "parquet"] | PixelFileFormatSpec = "parquet",
    ) -> None:
        """Save the PixelDataset to a .pxl file in the location provided in `path`.

        :param path: the path where to save the dataset as a .pxl
        :param file_format: should be 'csv' or 'parquet'. Default is 'parquet'.
                            This indicates what file-format is used to serialize
                            the data frames in the .pxl file.
        :returns: None
        :rtype: None
        :raises: AssertionError if invalid file format specified
        """
        logger.debug("Saving PixelDataset to %s", path)

        if isinstance(file_format, PixelFileFormatSpec):
            format_spec = file_format
        elif file_format not in ["csv", "parquet"]:
            raise AssertionError("`file_format` must be `csv` or `parquet`")
        if file_format == "csv":
            format_spec = PixelFileCSVFormatSpec()
        if file_format == "parquet":
            format_spec = PixelFileParquetFormatSpec()

        format_spec.save(self, path)

    def filter(
        self,
        components: Optional[pd.Series] = None,
        markers: Optional[pd.Series] = None,
    ) -> PixelDataset:
        """Filter the PixelDataset by components, markers, or both.

        Please note that markers will be filtered from the edgelist, even when provided,
        since it might cause different components to form, and hence invalidated the
        rest of the data.

        :param components: The components you want to keep, defaults to None
        :param markers: The markers you want to keep, defaults to None
        :return: A new instance of PixelDataset with the components/markers selected
        :rtype: PixelDataset
        """
        change_components = components is not None
        change_markers = markers is not None

        def _all_true_array(shape):
            return np.full(shape, True)

        adata_component_mask = (
            self.adata.obs.index.isin(components)
            if change_components
            else _all_true_array(self.adata.obs.index.shape)
        )
        adata_marker_mask = (
            self.adata.var.index.isin(markers)
            if change_markers
            else _all_true_array(self.adata.var.index.shape)
        )
        adata = self.adata[adata_component_mask, adata_marker_mask]
        update_metrics_anndata(adata, inplace=True)

        with pl.StringCache():
            lz_df = self.edgelist_lazy
            edgelist_pred = (
                lz_df.filter(pl.col("component").is_in(set(components)))  # type: ignore
                if change_components
                else self.edgelist_lazy
            )

            edgelist = _enforce_edgelist_types(edgelist_pred.collect().to_pandas())

        if self.polarization is not None:
            polarization_mask = (
                self.polarization["component"].isin(components)
                if change_components
                else _all_true_array(self.polarization.index.shape)
            ) & (
                self.polarization["marker"].isin(markers)
                if change_markers
                else _all_true_array(self.polarization.index.shape)
            )
            polarization = self.polarization[polarization_mask]

        if self.colocalization is not None:
            colocalization_mask = (
                self.colocalization["component"].isin(components)
                if change_components
                else _all_true_array(self.colocalization.index.shape)
            ) & (
                self.colocalization["marker_1"].isin(markers)
                & self.colocalization["marker_2"].isin(markers)
                if change_markers
                else _all_true_array(self.colocalization.index.shape)
            )
            colocalization = self.colocalization[colocalization_mask]

        return PixelDataset.from_data(
            adata=adata,
            edgelist=edgelist,
            metadata=self.metadata,
            polarization=polarization if self.polarization is not None else None,
            colocalization=colocalization if self.colocalization is not None else None,
        )
