"""Module for PixelDataset data stores, e.g. pxl files and similar.

Copyright (c) 2024 Pixelgen Technologies AB.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

import anndata as ad
import h5py
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from anndata import AnnData
from fsspec.implementations.zip import ZipFileSystem

if TYPE_CHECKING:
    from pixelator.pixeldataset import PixelDataset
from pixelator.types import PathType

logger = logging.getLogger(__name__)


@runtime_checkable
class PixelDataStore(Protocol):
    """Interface for pixel data storage.

    The methods here should be implemented by any pixel data store.

    """

    @staticmethod
    def from_path(path: PathType) -> PixelDataStore:
        """Get a PixelDataStore from the provided path.

        :param path: The path to the pixel data store.
        :return: A pixel data store.
        :rtype: PixelDataStore
        :raises ValueError: If the datastore format cannot be guessed from the path.
        """
        return PixelDataStore.guess_datastore_from_path(path)

    @staticmethod
    def guess_datastore_from_path(path: PathType) -> PixelDataStore:
        """Guess the pixel data store format based on the given path.

        :param path: The path to the pixel data store.
        :return: The guessed pixel data store format.
        :rtype: PixelDataStore
        :raises ValueError: If the datastore format cannot be guessed from the path.
        """
        if str(path).endswith(".pxl"):
            return ZipBasedPixelFile.guess_file_format(path)
        raise ValueError(f"Could not guess datastore from path {path}")

    def file_format_version(self) -> Optional[int]:
        """Return the file format version of the pixel data store."""
        ...

    def read_anndata(self) -> AnnData:
        """Read the pixel data as an AnnData object.

        :return: The pixel data as an AnnData object.
        :rtype: AnnData
        """
        ...

    def write_anndata(self, anndata: AnnData) -> None:
        """Write the given AnnData object to the pixel data store.

        :param anndata: The AnnData object to write.
        """
        ...

    def read_metadata(self) -> Dict[str, Any]:
        """Read the metadata associated with the pixel data store.

        :return: The metadata as a dictionary.
        :rtype: Dict[str, Any]
        """
        ...

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write the given metadata to the pixel data store.

        :param metadata: The metadata to write.
        """
        ...

    def read_dataframe(self, key: str) -> pd.DataFrame:
        """Read a dataframe from the pixel data store.

        :param key: The key of the dataframe to read.
        :return: The dataframe.
        :rtype: pd.DataFrame
        """
        ...

    def read_dataframe_lazy(self, key: str) -> pd.DataFrame:
        """Read a lazy dataframe from the pixel data store.

        :param key: The key of the dataframe to read.
        :return: The lazy dataframe.
        :rtype: pd.DataFrame
        """
        ...

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
        key: str,
        partitioning: Optional[list[str]] = None,
    ) -> None:
        """Write the given dataframe to the pixel data store.

        If a partitioning is provided, the dataframe will be partitioned accordingly.
        This may not be supported by all data stores. If not supported, this
        option will be ignored.

        :param dataframe: The dataframe to write.
        :param key: The key to write the dataframe to.
        :param partitioning: The (optional) partitioning to use when
                             writing the dataframe.
        """
        ...

    def read_edgelist(self) -> pd.DataFrame:
        """Read an edgelist from the pixel data store.

        :return: The edgelist.
        :rtype: pd.DataFrame
        """
        ...

    def read_edgelist_lazy(self) -> pd.DataFrame:
        """Read a lazy edgelist from the pixel data store.

        :return: The lazy edgelist.
        :rtype: pd.DataFrame
        """
        ...

    def read_polarization(self) -> Optional[pd.DataFrame]:
        """Read the polarization data from the pixel data store.

        :return: The polarization data, or None if it doesn't exist.
        :rtype: Optional[pd.DataFrame]
        """
        ...

    def read_colocalization(self) -> Optional[pd.DataFrame]:
        """Read the colocalization data from the pixel data store.

        :return: The colocalization data, or None if it doesn't exist.
        :rtype: Optional[pd.DataFrame]
        """
        ...

    def write_edgelist(
        self, edgelist: pd.DataFrame, partitioning: Optional[list[str]] = None
    ) -> None:
        """Write the given edgelist to the pixel data store.

        If a partitioning is provided, the edgelist will be partitioned accordingly.
        This may not be supported by all data stores. If not supported, this
        option will be ignored.

        :param edgelist: The edgelist to write.
        :param partitioning: The (optional) partitioning to use when
                             writing the dataframe.
        """
        ...

    def write_polarization(self, polarization: pd.DataFrame) -> None:
        """Write the given polarization data to the pixel data store.

        :param polarization: The polarization data to write.
        """
        ...

    def write_colocalization(self, colocalization: pd.DataFrame) -> None:
        """Write the given colocalization data to the pixel data store.

        :param colocalization: The colocalization data to write.
        """
        ...

    def save(self, dataset: PixelDataset) -> None:
        """Save the given PixelDataset to the pixel data store.

        :param dataset: The PixelDataset to save.
        """
        ...


class PixelDataStoreError(Exception):
    """Base class for all PixelDataStore related exceptions."""

    pass


class FileFormatNotRecognizedError(PixelDataStoreError):
    """Raised when the file format of a .pxl file is not recognized."""

    pass


class EdgelistNotFoundError(PixelDataStoreError):
    """Raised when the edgelist is not found in a .pxl file."""

    pass


class ZipBasedPixelFile(PixelDataStore):
    """Superclass for all zip-based pixel data stores."""

    FILE_FORMAT_VERSION: int = 1

    FILE_FORMAT_VERSION_KEY: str = "file_format_version"
    ANNDATA_KEY: str
    EDGELIST_KEY: str
    METADATA_KEY: str
    POLARIZATION_KEY: str
    COLOCALIZATION_KEY: str

    def __init__(self, path: PathType) -> None:
        """Create a zip-based pixel data store."""
        self.path = path
        self._file_system_handle = None
        self._current_mode: Literal["r", "w", "a"] | None = None

    def __del__(self) -> None:
        """Ensure file handle is closed when object is deleted."""
        # Make sure the file handles are closed when the object is deleted
        if self._file_system_handle:
            self._file_system_handle.close()

    def _setup_file_system(self, mode):
        self._file_system_handle = ZipFileSystem(fo=self.path, mode=mode)
        self._current_mode = mode

    @property
    def _file_system(self):
        if not self._file_system_handle:
            self._setup_file_system("r")
        return self._file_system_handle

    def _set_to_write_mode(self):
        if not (self._current_mode == "w" or self._current_mode == "a"):
            if self._file_system_handle:
                self._file_system_handle.close()
            if Path(self.path).is_file():
                self._setup_file_system("a")
            else:
                self._setup_file_system("w")

    def _set_to_read_mode(self):
        if self._current_mode != "r":
            if self._file_system_handle:
                self._file_system_handle.close()
            self._setup_file_system("r")

    @staticmethod
    def from_file(path) -> PixelDataStore:
        """Guess the file format of the given path and returns the a PixelDataStore."""
        return ZipBasedPixelFile.guess_file_format(path)

    @staticmethod
    def guess_file_format(path: PathType) -> PixelDataStore:
        """Guess the file format of the given path and returns the a PixelDataStore."""
        file_system = ZipFileSystem(fo=path, mode="r")
        members = [file_["name"] for file_ in file_system.ls("/")]
        file_format: Optional[Type[ZipBasedPixelFile]] = None
        if ZipBasedPixelFileWithParquet.EDGELIST_KEY in members:
            file_format = ZipBasedPixelFileWithParquet
        if ZipBasedPixelFileWithCSV.EDGELIST_KEY in members:
            file_format = ZipBasedPixelFileWithCSV
        if not file_format:
            raise FileFormatNotRecognizedError(
                f"Could not identify file format of input file {path}"
            )
        file_system.close()
        return file_format(path)

    def file_format_version(self) -> Optional[int]:
        """Return the file format version of the .pxl file."""
        self._set_to_read_mode()
        return self.read_metadata().get(self.FILE_FORMAT_VERSION_KEY)

    def read_anndata(self) -> AnnData:
        """Read the AnnData object from the .pxl file."""
        self._set_to_read_mode()
        with self._file_system.open(self.ANNDATA_KEY, "rb") as af:
            with h5py.File(af, "r") as f:
                data = ad.experimental.read_elem(f["/"])
        return data

    def write_anndata(self, anndata: AnnData) -> None:
        """Write the given AnnData object to the .pxl file."""
        self._set_to_write_mode()
        with self._file_system.open(self.ANNDATA_KEY, "wb", compression=None) as af:
            bio = io.BytesIO()
            with h5py.File(bio, "w") as f:
                ad.experimental.write_elem(f, "/", anndata)
            af.write(bio.getbuffer())

    def read_edgelist(self) -> pd.DataFrame:
        """Read the edgelist from the .pxl file."""
        self._set_to_read_mode()
        df = self.read_dataframe(self.EDGELIST_KEY)
        if df is None:
            raise EdgelistNotFoundError("Edgelist not found in pxl file")
        return df

    def read_edgelist_lazy(self) -> Optional[pl.LazyFrame]:
        """Read the edgelist lazily from the .pxl file."""
        self._set_to_read_mode()
        return self.read_dataframe_lazy(self.EDGELIST_KEY)

    def read_polarization(self) -> Optional[pd.DataFrame]:
        """Read the polarization data from the .pxl file."""
        self._set_to_read_mode()
        return self.read_dataframe(self.POLARIZATION_KEY)

    def read_colocalization(self) -> Optional[pd.DataFrame]:
        """Read the colocalization data from the .pxl file."""
        self._set_to_read_mode()
        return self.read_dataframe(self.COLOCALIZATION_KEY)

    def read_metadata(self) -> Dict:
        """Read the metadata from the .pxl file."""
        self._set_to_read_mode()
        try:
            with self._file_system.open(self.METADATA_KEY, "r") as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return {}

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write the given metadata to the .pxl file."""
        self._set_to_write_mode()
        with self._file_system.open(self.METADATA_KEY, "w") as f:
            f.write(json.dumps(metadata))

    def write_edgelist(
        self, edgelist: pd.DataFrame, partitioning: Optional[list[str]] = None
    ) -> None:
        """Write the given edgelist to the .pxl file."""
        self._set_to_write_mode()
        self.write_dataframe(edgelist, self.EDGELIST_KEY, partitioning)

    def write_polarization(self, polarization: pd.DataFrame) -> None:
        """Write the given polarization data to the .pxl file."""
        self._set_to_write_mode()
        self.write_dataframe(polarization, self.POLARIZATION_KEY)

    def write_colocalization(self, colocalization: pd.DataFrame) -> None:
        """Write the given colocalization data to the .pxl file."""
        self._set_to_write_mode()
        self.write_dataframe(colocalization, self.COLOCALIZATION_KEY)

    def save(self, dataset: PixelDataset) -> None:
        """Save the given PixelDataset to the .pxl file."""
        self._set_to_write_mode()
        self.write_anndata(dataset.adata)
        # TODO Consider using default partitioning for edgelist here
        self.write_edgelist(dataset.edgelist)
        self.write_metadata(dataset.metadata)
        if dataset.polarization is not None and dataset.polarization.shape[0] > 0:
            self.write_polarization(dataset.polarization)

        if dataset.colocalization is not None and dataset.colocalization.shape[0] > 0:
            self.write_colocalization(dataset.colocalization)

        logger.debug("PixelDataset saved to %s", self.path)


class ZipBasedPixelFileWithCSV(ZipBasedPixelFile):
    """A zip-based pixel data store that uses csv files for data storage.

    This is only provided for backward compatibility, the ZipBasedPixelFileWithParquet
    should be used instead of this for all new code.
    """

    ANNDATA_KEY: str = "adata.h5ad"
    EDGELIST_KEY: str = "edgelist.csv.gz"
    METADATA_KEY: str = "metadata.json"
    POLARIZATION_KEY: str = "polarization.csv.gz"
    COLOCALIZATION_KEY: str = "colocalization.csv.gz"

    def __init__(self, path: PathType) -> None:
        """Create a zip-based pixel file using csv files to store dataframes."""
        super().__init__(path)

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
        key: str,
        partitioning: Optional[list[str]] = None,
    ) -> None:
        """Write the given dataframe to the .pxl file."""
        # Note that partitioning will be ignored here
        with self._file_system.open(key, "wb", compression=None) as f:
            dataframe.to_csv(f, compression="gzip", index=False)

    def read_dataframe(self, key: PathType) -> Optional[pd.DataFrame]:
        """Read a dataframe from the .pxl file."""
        return self._read_dataframe_from_zip(key)

    def read_dataframe_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        """Read a dataframe lazily from a zip file (NB: Not implemented!)."""
        raise NotImplementedError(
            "You are trying to read data lazily from a csv-based pxl file. "
            "This is currently not supported. "
            "You can fix this issue by converting your pxl file by saving it "
            "as a parquet based pxl file."
        )

    def _read_dataframe_from_zip(self, key: PathType) -> Optional[pd.DataFrame]:
        try:
            with self._file_system.open(key, "rb") as f:
                return pl.read_csv(f).to_pandas()
        except KeyError:
            return None


class ZipBasedPixelFileWithParquet(ZipBasedPixelFile):
    """A zip-based pixel data store that uses parquet files for data storage."""

    ANNDATA_KEY: str = "adata.h5ad"
    EDGELIST_KEY: str = "edgelist.parquet"
    METADATA_KEY: str = "metadata.json"
    POLARIZATION_KEY: str = "polarization.parquet"
    COLOCALIZATION_KEY: str = "colocalization.parquet"

    def __init__(self, path) -> None:
        """Create a zip-based pixel file using parquet files to store dataframes."""
        super().__init__(path)

    def write_dataframe(
        self,
        dataframe: pd.DataFrame,
        key: str,
        partitioning: Optional[list[str]] = None,
    ) -> None:
        """Write the given dataframe to the .pxl file.

        Optionally provided a partitioning to create a hive partitioned parquet file.
        I.e. with a separate file for each level of partitioning provided.

        :param dataframe: The dataframe to write.
        :param key: The key of the dataframe to write.
        :param partitioning: The partitioning to use when writing the dataframe.
        """
        if partitioning:
            for _, data in dataframe.groupby(partitioning):
                ds.write_dataset(
                    pa.Table.from_pandas(data),
                    f"{key}/",
                    filesystem=self._file_system,
                    format="parquet",
                    partitioning_flavor="hive",
                    partitioning=partitioning,
                    use_threads=False,
                    existing_data_behavior="overwrite_or_ignore",
                )
            return

        pq.write_table(
            pa.Table.from_pandas(dataframe), where=key, filesystem=self._file_system
        )

    def read_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Read a dataframe from the .pxl file."""
        return self._read_dataframe_from_zip(key)

    def read_dataframe_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        """Read a dataframe lazily from a zip file."""
        return self._read_dataframe_from_zip_lazy(key)

    def _read_dataframe_from_zip(self, key: str) -> Optional[pd.DataFrame]:
        df = self._read_dataframe_from_zip_lazy(key)
        if df is None:
            return None
        return df.collect().to_pandas()

    def _read_dataframe_from_zip_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        try:
            dataset = ds.dataset(key, filesystem=self._file_system)
            return pl.scan_pyarrow_dataset(dataset)
        except FileNotFoundError:
            return None
