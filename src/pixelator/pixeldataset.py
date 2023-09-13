"""Module for PixelDataset and associated functions.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
)
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData, read_h5ad
from anndata import concat as concatenate_anndata

from pixelator.graph import Graph, components_metrics
from pixelator.statistics import (
    clr_transformation,
    denoise,
    log1p_transformation,
    rel_normalization,
)
from pixelator.types import PathType

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", module="libpysal")

if TYPE_CHECKING:
    from pixelator.config import AntibodyPanel

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


def simple_aggregate(
    sample_names: List[str], datasets: List[PixelDataset]
) -> PixelDataset:
    """Aggregate samples in a simple way (see caveats).

    Aggregating samples in a simplistic fashion. This function should only
    be used if the dataset you merge have been generated with the same panel.

    It will concatenate all dataframes in the underlying PixelDataset instances,
    and add a new column called sample. New indexes will be formed from from the
    `sample` and `component` columns.

    The metadata dictionary will contain one key per sample.

    :param sample_names: an iterable of the sample names to use for each dataset
    :param datasets: an iterable of the datasets you want to aggregate
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

    def _get_attr_and_index_by_component(attribute):
        for name, dataset in zip(sample_names, datasets):
            attr = getattr(dataset, attribute, None)
            if attr is not None:
                attr["sample"] = name
                attr["component"] = attr["component"].astype(str) + "_" + attr["sample"]
                attr.set_index(["component"])
                yield attr

    edgelists = pd.concat(_get_attr_and_index_by_component("edgelist"), axis=0)
    polarizations = pd.concat(_get_attr_and_index_by_component("polarization"), axis=0)
    colocalizations = pd.concat(
        _get_attr_and_index_by_component("colocalization"), axis=0
    )
    metadata = {
        "samples": {
            name: dataset.metadata for name, dataset in zip(sample_names, datasets)
        }
    }

    return PixelDataset.from_data(
        adata=adata,
        edgelist=edgelists,
        polarization=polarizations,
        colocalization=colocalizations,
        metadata=metadata,
    )


class PixelDatasetBackend(Protocol):
    """Protocol for PixelDataset backends.

    This protocol defines the methods required by a PixelDataset backend.
    Any class that fulfills this contract can be used as a backend for a
    PixelDataset.

    Normally the backend used by pixelator is a .pxl file, but by implementing
    this protocol you can provide other backends, to for example load data directly
    from a database.
    """

    @property
    def adata(self) -> AnnData:
        """Return the AnnData instance."""

    @adata.setter
    def adata(self, value: AnnData) -> None:
        """Set the AnnData instance."""

    @property
    def edgelist(self) -> pd.DataFrame:
        """Return the edgelist."""

    @edgelist.setter
    def edgelist(self, value: pd.DataFrame) -> None:
        """Set the edge list instance."""

    @property
    def polarization(self) -> Optional[pd.DataFrame]:
        """Return the polarization data."""

    @polarization.setter
    def polarization(self, value: pd.DataFrame) -> None:
        """Set the polarization data frame."""

    @property
    def colocalization(self) -> Optional[pd.DataFrame]:
        """Return the colocalization data."""

    @colocalization.setter
    def colocalization(self, value: pd.DataFrame) -> None:
        """Set the colocalization data frame."""

    @property
    def metadata(self) -> Optional[Dict]:
        """Return the metadata object."""

    @metadata.setter
    def metadata(self, value: Dict) -> Optional[Dict]:
        """Set the metadata object."""


class PixelFileFormatSpec(ABC):
    """Abstract base class for PixelFileFormatSpecs.

    An abstract base class used to specify how to load and save a PixelDataset
    instance from disk. It assumes that the pixel file is a zip archive when saving
    a PixelDataset. To implement other data backends see `PixelDatasetBackend`
    """

    FILE_FORMAT_VERSION: int = 1

    FILE_FORMAT_VERSION_KEY: str = "file_format_version"
    ANNDATA_KEY: str
    EDGELIST_KEY: str
    METADATA_KEY: str
    POLARIZATION_KEY: str
    COLOCALIZATION_KEY: str

    @staticmethod
    def guess_file_format(path: PathType) -> PixelFileFormatSpec:
        """Attempt to guess and the file format of the .pxl file in `path`.

        :param path: path to a .pxl file
        :raises AssertionError: If any mandatory information is missing in the file
        :return: A `PixelFileFormatSpec` instance that can be used to access
                 the data in the file
        :rtype: PixelFileFormatSpec
        """
        with ZipFile(path, "r") as zip_archive:
            members = zip_archive.namelist()
            if PixelFileParquetFormatSpec.EDGELIST_KEY in members:
                file_format: Type[PixelFileFormatSpec] = PixelFileParquetFormatSpec
            else:
                file_format = PixelFileCSVFormatSpec

            if file_format.ANNDATA_KEY not in members:
                raise AssertionError(
                    f"Input dataset {path} is missing the mandatory adata object"
                )

            if file_format.EDGELIST_KEY not in members:
                raise AssertionError(
                    f"Input dataset {path} is missing the mandatory edge list object"
                )

            return file_format()

    @staticmethod
    @abstractmethod
    def serialize_dataframe(dataframe: pd.DataFrame, path: PathType) -> None:
        """Serialize a data frame.

        Serialize a data frame to the location indicated by `path`

        :param dataframe: a pd.DataFrame to serialize
        :param path: the path to write the data frame to
        """
        ...

    @staticmethod
    @abstractmethod
    def deserialize_dataframe(path: PathType, key: str) -> pd.DataFrame:
        """Deserialize a data frame from a .pxl file.

        Deserialize a data frame from a .pxl file at `path`, which has been
        stored with the file name `key` in the zip archive

        :param path: the path to a pxl file
        :param key: the file name that the data frame was stored under
        :return: a `pd.DataFrame` instance
        :rtype: pd.DataFrame
        """
        ...

    def file_format_version(self, path: PathType) -> Optional[int]:
        """Get the file format version of the file given `path`.

        :param path: the path to a pxl file
        :return: the file format version of the pixel file
        :rtype: Optional[int]
        """
        return self.deserialize_metadata(path).get(self.FILE_FORMAT_VERSION_KEY)

    def deserialize_anndata(self, path: PathType) -> AnnData:
        """Deserialize an `AnnData` instance from a .pxl file at `path`.

        :param path: the path to a pxl file
        :return: an `AnnData` instance
        :rtype: AnnData
        """
        with ZipFile(path, "r") as zip_archive:
            with tempfile.TemporaryDirectory() as tmp_dir:
                adata_file = zip_archive.extract(self.ANNDATA_KEY, tmp_dir)
                return read_anndata(adata_file)

    def deserialize_metadata(self, path: PathType) -> Dict:
        """Deserialize a metadata file from a .pxl file.

        :param path: path to the .pxl file
        :return: A dictionary containing the metadata
        :rtype: Dict
        """
        with ZipFile(path, "r") as zip_archive:
            members = zip_archive.namelist()
            if self.METADATA_KEY not in members:
                return {}
            with zip_archive.open(self.METADATA_KEY) as f:
                return json.loads(f.read())

    def save(self, dataset: PixelDataset, path: PathType) -> None:
        """Save the given a `PixelDataset` to the `path` provided as a .pxl file.

        :param dataset: a PixelDataset instance to save
        :param path: the path to save the file to
        """
        # TODO Avoid writing temporary file to disk as much as possible here
        # The current implementation create temporary files which are then
        # written into the zip archive.
        with ZipFile(path, "w", compression=ZIP_STORED) as zip_archive:
            # create and save temp AnnData file
            file = tempfile.mkstemp(suffix=".h5ad")[1]
            write_anndata(dataset.adata, file)
            zip_archive.write(file, self.ANNDATA_KEY)
            Path(file).unlink()

            # create and save temp edge list file
            file = tempfile.mkstemp(suffix=".csv")[1]
            self.serialize_dataframe(dataset.edgelist, file)
            zip_archive.write(file, self.EDGELIST_KEY)
            Path(file).unlink()

            if dataset.metadata is not None:
                # save metadata as JSON
                metadata = dataset.metadata.copy()
                metadata = {
                    **metadata,
                    **{self.FILE_FORMAT_VERSION_KEY: self.FILE_FORMAT_VERSION},
                }
                zip_archive.writestr(self.METADATA_KEY, json.dumps(metadata))

            if dataset.polarization is not None:
                # create and save temporary polarization scores
                file = tempfile.mkstemp(suffix=".csv.gz")[1]
                self.serialize_dataframe(dataset.polarization, file)
                zip_archive.write(file, self.POLARIZATION_KEY)
                Path(file).unlink()

            if dataset.colocalization is not None:
                # create and save temporary colocalization scores
                file = tempfile.mkstemp(suffix=".csv.gz")[1]
                self.serialize_dataframe(dataset.colocalization, file)
                zip_archive.write(file, self.COLOCALIZATION_KEY)
                Path(file).unlink()

        logger.debug("PixelDataset saved to %s", path)


class PixelFileCSVFormatSpec(PixelFileFormatSpec):
    """Format specification for pxl-files using csv as on-disk storage.

    A format specification for .pxl files that store their dataframes as
    csv files. This is mostly used for backwards compatibility.
    """

    ANNDATA_KEY: str = "adata.h5ad"
    EDGELIST_KEY: str = "edgelist.csv.gz"
    METADATA_KEY: str = "metadata.json"
    POLARIZATION_KEY: str = "polarization.csv.gz"
    COLOCALIZATION_KEY: str = "colocalization.csv.gz"

    @staticmethod
    def serialize_dataframe(dataframe: pd.DataFrame, path: PathType) -> None:
        """Serialize a dataframe from the give path."""
        dataframe.to_csv(path, header=True, index=False, compression="gzip")

    @staticmethod
    def deserialize_dataframe(path: PathType, key: PathType) -> pd.DataFrame:
        """Deserialize a dataframe from the give path."""
        return PixelFileCSVFormatSpec._read_dataframe_from_zip(path, key)

    @staticmethod
    def _read_dataframe_from_zip(
        path: PathType, key: PathType
    ) -> Optional[pd.DataFrame]:
        with ZipFile(path, "r") as zip_archive:
            members = zip_archive.namelist()
            if key not in members:
                return None
            with zip_archive.open(key) as f:  # type: ignore
                with gzip.open(f, mode="rb") as gz:
                    # A hack to get polars to skip the warning:
                    # "Polars found a filename. Ensure you pass a path to the file
                    # instead of a python file object when possible for best
                    # performance."
                    delattr(gz, "name")
                    return pl.read_csv(gz).to_pandas()  # type: ignore


class PixelFileParquetFormatSpec(PixelFileFormatSpec):
    """Format specification for pxl-files using parquet as on-disk storage.

    A format specification for .pxl files that store their dataframes as
    parquet files. This is the current default file format.
    """

    ANNDATA_KEY: str = "adata.h5ad"
    EDGELIST_KEY: str = "edgelist.parquet"
    METADATA_KEY: str = "metadata.json"
    POLARIZATION_KEY: str = "polarization.parquet"
    COLOCALIZATION_KEY: str = "colocalization.parquet"

    @staticmethod
    def serialize_dataframe(dataframe: pd.DataFrame, path: PathType) -> None:
        """Serialize a dataframe from the give path."""
        kwargs = {"compression": "zstd"}
        pl.from_pandas(dataframe).write_parquet(path, **kwargs)  # type: ignore

    @staticmethod
    def deserialize_dataframe(path: PathType, key: str) -> pd.DataFrame:
        """Deserialize a dataframe from the give path."""
        return PixelFileParquetFormatSpec._read_dataframe_from_zip(path, key)

    @staticmethod
    def _read_dataframe_from_zip(path: PathType, key: str) -> Optional[pd.DataFrame]:
        with ZipFile(path, "r") as zip_archive:
            members = zip_archive.namelist()
            if key not in members:
                return None
            with zip_archive.open(key) as f:
                df = pl.read_parquet(f)  # type: ignore
                return df.to_pandas(use_pyarrow_extension_array=True)


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
        edgelist: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        polarization: Optional[pd.DataFrame] = None,
        colocalization: Optional[pd.DataFrame] = None,
    ) -> PixelDataset:
        """Create a new instance of PixelDataset from the provided underlying objects.

        :param adata: an instance of `AnnData`
        :param edgelist: an edgelist as a `pd.DataFrame`
        :param metadata: an instance of a dictionary with metadata, defaults to None
        :param polarization: a `pd.DataFrame` with polarization information,
                             defaults to None
        :param colocalization: a `pd.DataFrame` with colocalization information,
                               defaults to None
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
        return self._backend.edgelist

    @edgelist.setter
    def edgelist(self, value: pd.DataFrame) -> None:
        """Set the edge list."""
        self._backend.edgelist = value

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
            if not np.any(self.edgelist["component"] == component_id):
                raise KeyError(f"{component_id} not found in edgelist")
            return Graph.from_edgelist(
                self.edgelist[self.edgelist["component"] == component_id],
                add_marker_counts=add_node_marker_counts,
                simplify=simplify,
                use_full_bipartite=use_full_bipartite,
            )
        return Graph.from_edgelist(
            self.edgelist,
            add_marker_counts=add_node_marker_counts,
            simplify=simplify,
            use_full_bipartite=use_full_bipartite,
        )

    def __str__(self) -> str:
        """Get a string representation of this object."""
        msg = (
            f"Pixel dataset contains:\n"
            f"\tAnnData with {self.adata.n_obs} obs and {self.adata.n_vars} vars\n"
            f"\tEdge list with {self.edgelist.shape[0]} edges"
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

        edgelist_mask = (
            self.edgelist["component"].isin(components)
            if change_components
            else _all_true_array(self.edgelist.index.shape)
        )
        edgelist = self.edgelist[edgelist_mask]

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


class ObjectBasedPixelDatasetBackend:
    """A backend for PixelDataset that is backed by in memory objects.

    `ObjectBasedPixelDatasetBackend` provides a backend for PixelDatasets that
    are stored entirely in memory. This is mostly used by Pixelator internally
    the first time the PixelDataset instance is created.

    Note that it will make it's own copy of all the provided input data.
    """

    def __init__(
        self,
        adata: AnnData,
        edgelist: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        polarization: Optional[pd.DataFrame] = None,
        colocalization: Optional[pd.DataFrame] = None,
    ) -> None:
        """Create a new instance of ObjectBasedPixelDatasetBackend.

        :param adata: an AnnData instance
        :param edgelist: an edgelist dataframe
        :param metadata: a dict with metadata, defaults to None
        :param polarization: a polarization dataframe, defaults to None
        :param colocalization: a colocalization dataframe, defaults to None
        :raises AssertionError: if `adata` or `edgelist` contains no data.
        """
        if adata is None or adata.n_obs == 0:
            raise AssertionError("adata cannot be empty")

        if edgelist is None or edgelist.shape[0] == 0:
            raise AssertionError("edgelist cannot be empty")

        self._edgelist = edgelist.copy()
        self._adata = adata.copy()
        self._metadata = metadata
        self._polarization = None
        if polarization is not None:
            self._polarization = polarization.copy()
        self._colocalization = None
        if colocalization is not None:
            self._colocalization = colocalization.copy()

    @property
    def adata(self) -> AnnData:
        """Get the AnnData object for the pixel dataset."""
        return self._adata

    @adata.setter
    def adata(self, value: AnnData) -> None:
        """Set the AnnData object for the pixel dataset."""
        self._adata = value

    @property
    def edgelist(self) -> pd.DataFrame:
        """Get the edge list for the pixel dataset."""
        return self._edgelist

    @edgelist.setter
    def edgelist(self, value: pd.DataFrame) -> None:
        """Set the edge list for the pixel dataset."""
        self._edgelist = value

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Get the metadata for the pixel dataset."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set the metadata for the pixel dataset."""
        self._metadata = value

    @property
    def polarization(self) -> Optional[pd.DataFrame]:
        """Get the polarization scores for the pixel dataset."""
        return self._polarization

    @polarization.setter
    def polarization(self, value: pd.DataFrame) -> None:
        """Set the polarization scores for the pixel dataset."""
        self._polarization = value

    @property
    def colocalization(self) -> Optional[pd.DataFrame]:
        """Get the co-localization scores for the pixel dataset."""
        return self._colocalization

    @colocalization.setter
    def colocalization(self, value: pd.DataFrame) -> None:
        """Set the co-localization scores for the pixel dataset."""
        self._colocalization = value


class FileBasedPixelDatasetBackend:
    """A file based backend for PixelDataset.

    `FileBasedPixelDatasetBackend` is used to lazily fetch information from
    a .pxl file. Once the file has been read the data will be cached
    in memory.
    """

    def __init__(self, path: PathType) -> None:
        """Create a filebased backend instance.

        Create a backend, fetching information from the .pxl file
        in `path`.

        :param path: Path to the .pxl file
        """
        self._path = path
        self._file_format = PixelFileFormatSpec.guess_file_format(path)

    @cached_property
    def adata(self) -> AnnData:
        """Get the AnnData object for the pixel dataset."""
        return self._file_format.deserialize_anndata(self._path)

    @cached_property
    def edgelist(self) -> pd.DataFrame:
        """Get the edge list object for the pixel dataset."""
        return self._file_format.deserialize_dataframe(
            self._path, self._file_format.EDGELIST_KEY
        )

    @cached_property
    def polarization(self) -> Optional[pd.DataFrame]:
        """Get the polarization object for the pixel dataset."""
        return self._file_format.deserialize_dataframe(
            self._path, self._file_format.POLARIZATION_KEY
        )

    @cached_property
    def colocalization(self) -> Optional[pd.DataFrame]:
        """Get the colocalization object for the pixel dataset."""
        return self._file_format.deserialize_dataframe(
            self._path, self._file_format.COLOCALIZATION_KEY
        )

    @cached_property
    def metadata(self) -> Optional[Dict]:
        """Get the metadata object for the pixel dataset."""
        return self._file_format.deserialize_metadata(self._path)


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
        edgelist.groupby("marker")
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
        edgelist.groupby(["component", "marker"]).agg("size").unstack().fillna(0)
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
    .obsm["normalized_rel"] = the normalized (REL) component to antibody counts
    .obsm["clr"] = the transformed (clr) component to antibody counts
    .obsm["log1p"] = the transformed (log1p) component to antibody counts
    .obsm["denoised"] = the denoised (clr) counts if control antibodies are present

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
    counts_df_norm = rel_normalization(df=counts_df, axis=1)
    counts_df_clr = clr_transformation(df=counts_df, axis=1)
    counts_df_log1p = log1p_transformation(df=counts_df)
    adata.obsm["normalized_rel"] = counts_df_norm
    adata.obsm["clr"] = counts_df_clr
    adata.obsm["log1p"] = counts_df_log1p
    antibody_control = panel.markers_control
    if antibody_control is not None and len(antibody_control) > 0:
        adata.obsm["denoised"] = denoise(
            df=counts_df_clr,
            antibody_control=antibody_control,
            quantile=1.0,
            axis=1,
        )

    logger.debug("AnnData created")
    return adata


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
    adata.var["antibody_count"] = df.sum()
    adata.var["components"] = (df != 0).sum()
    adata.var["antibody_pct"] = (
        adata.var["antibody_count"] / adata.var["antibody_count"].sum()
    )

    # update the obs layer (components metrics)
    adata.obs["antibodies"] = np.sum(adata.X > 0, axis=1)

    logger.debug("Metrics in AnnData object updated")
    return None if inplace else adata
