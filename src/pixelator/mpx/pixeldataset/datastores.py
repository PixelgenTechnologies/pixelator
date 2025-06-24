"""Module for PixelDataset data stores, e.g. pxl files and similar.

Copyright © 2024 Pixelgen Technologies AB.
"""

from __future__ import annotations

import io
import json
import logging
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
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

from pixelator.common.exceptions import PixelatorBaseException
from pixelator.common.utils import batched
from pixelator.mpx.pixeldataset.precomputed_layouts import PreComputedLayouts

if TYPE_CHECKING:
    from pixelator.mpx.pixeldataset import PixelDataset

from pixelator.common.types import PathType

logger = logging.getLogger(__name__)


class PixelDataStoreError(PixelatorBaseException):
    """Base class for all PixelDataStore related exceptions."""

    pass


class CannotOverwriteError(PixelDataStoreError):
    """Raised when a file is attempted to be overwritten."""

    pass


class FileFormatNotRecognizedError(PixelDataStoreError):
    """Raised when the file format of a .pxl file is not recognized."""

    pass


class EdgelistNotFoundError(PixelDataStoreError):
    """Raised when the edgelist is not found in a .pxl file."""

    pass


class CannotGuessPixelDataStoreError(PixelDataStoreError):
    """Raised when the pixel data store format cannot be guessed from the path."""

    pass


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
        :raises CannotGuessPixelDatastoreError: If the datastore format cannot be guessed from the path.
        """
        return PixelDataStore.guess_datastore_from_path(path)

    @staticmethod
    def guess_datastore_from_path(path: PathType) -> PixelDataStore:
        """Guess the pixel data store format based on the given path.

        :param path: The path to the pixel data store.
        :return: The guessed pixel data store format.
        :rtype: PixelDataStore
        :raises CannotGuessPixelDatastoreError: If the datastore format cannot be guessed from the path.
        """
        if str(path).endswith(".pxl"):
            return ZipBasedPixelFile.guess_file_format(path)
        raise CannotGuessPixelDataStoreError(
            f"Could not guess datastore from path {path}"
        )

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

    def read_dataframe_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        """Read a lazy dataframe from the pixel data store.

        :param key: The key of the dataframe to read.
        :return: The lazy dataframe.
        :rtype: Optional[pl.LazyFrame]
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

    def read_edgelist_lazy(self) -> pl.LazyFrame:
        """Read a lazy edgelist from the pixel data store.

        :return: The lazy edgelist.
        :rtype: pl.LazyFrame
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

    def save(self, dataset: PixelDataset, force_overwrite: bool = False) -> None:
        """Save the given PixelDataset to the pixel data store.

        :param dataset: The PixelDataset to save.
        :param force_overwrite: Whether to force overwrite an existing file.
        """
        ...

    def read_precomputed_layouts(
        self,
    ) -> PreComputedLayouts:
        """Read pre-computed layouts from the data store."""
        ...

    def write_precomputed_layouts(
        self,
        layouts: PreComputedLayouts,
    ) -> None:
        """Write pre-computed layouts to the data store.

        :param layouts: The pre-computed layouts to write.
        :param collapse_to_single_dataframe: Whether to collapse the layouts into
                                             a single dataframe before writing.
        """
        ...


class _CustomZipFileSystem(ZipFileSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        result = {}

        def _below_max_recursion_depth(path):
            if not maxdepth:
                return True

            depth = len(path.split("/"))
            return depth <= maxdepth

        for zip_info in self.zip.infolist():
            file_name = zip_info.filename
            if not file_name.startswith(path.lstrip("/")):
                continue

            # zip files can contain explicit or implicit directories
            # hence the need to either add them directly or infer them
            # from the file paths
            if zip_info.is_dir():
                if withdirs:
                    if not result.get(file_name) and _below_max_recursion_depth(
                        file_name
                    ):
                        result[file_name.strip("/")] = (
                            self.info(file_name) if detail else None
                        )
                    continue
                else:
                    continue  # Skip along to the next entry if we don't want to add the dirs

            if not result.get(file_name):
                if _below_max_recursion_depth(file_name):
                    result[file_name] = self.info(file_name) if detail else None

                # Here we handle the case of implicitly adding the
                # directories if they have been requested
                if withdirs:
                    directories = file_name.split("/")
                    for i in range(1, len(directories)):
                        dir_path = "/".join(directories[:i]).strip(
                            "/"
                        )  # remove the trailing slash, as this is not expected
                        if not result.get(dir_path) and _below_max_recursion_depth(
                            dir_path
                        ):
                            result[dir_path] = {
                                "name": dir_path,
                                "size": 0,
                                "type": "directory",
                            }

        return result if detail else sorted(list(result.keys()))


class ZipBasedPixelFile(PixelDataStore):
    """Superclass for all zip-based pixel data stores."""

    FILE_FORMAT_VERSION: int = 1

    FILE_FORMAT_VERSION_KEY: str = "file_format_version"
    ANNDATA_KEY: str
    COLOCALIZATION_KEY: str
    EDGELIST_KEY: str
    METADATA_KEY: str
    LAYOUTS_KEY: str = "layouts.parquet"
    POLARIZATION_KEY: str

    def __init__(self, path: PathType) -> None:
        """Create a zip-based pixel data store."""
        self.path = path
        self._file_system_handle = None
        self._current_mode: Literal["r", "w", "a"] | None = None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()

    def __del__(self) -> None:
        """Ensure file handle is closed when object is deleted."""
        # Make sure the file handles are closed when the object is deleted
        self.close()

    def _setup_file_system(self, mode):
        files_system = _CustomZipFileSystem(fo=self.path, mode=mode, allowZip64=True)

        # For now we are overwriting the zip open method to force it to
        # always have force_zip=True, otherwise it won't work for large file
        # This is related to this issue in fsspec:
        # https://github.com/fsspec/filesystem_spec/issues/1474
        def custom_open(f):
            return partial(f, force_zip64=True)

        files_system.zip.open = custom_open(files_system.zip.open)
        self._file_system_handle = files_system
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

    def _check_if_writeable(self, key):
        try:
            if self._file_system.exists(key):
                raise CannotOverwriteError(
                    f"Data with key {key} already exists in pixel file. We currently do not "
                    "support overwriting. Please save your .pxl file under a new name."
                )
        except FileNotFoundError:
            return True

    def close(self):
        """Close the file system handle."""
        if self._file_system_handle:
            self._file_system_handle.close()

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
        self._check_if_writeable(self.ANNDATA_KEY)
        with self._file_system.open(self.ANNDATA_KEY, "wb", compression=None) as af:
            bio = io.BytesIO()
            with h5py.File(bio, "w") as f:
                ad.experimental.write_elem(f, "/", anndata)
            af.write(bio.getbuffer())

    def read_edgelist(self) -> pd.DataFrame:
        """Read the edgelist from the .pxl file."""
        df = self.read_dataframe(self.EDGELIST_KEY)
        if df is None:
            raise EdgelistNotFoundError("Edgelist not found in pxl file")
        return df

    def read_edgelist_lazy(self) -> pl.LazyFrame:
        """Read the edgelist lazily from the .pxl file."""
        lazy_data_frame = self.read_dataframe_lazy(self.EDGELIST_KEY)
        if lazy_data_frame is None:
            raise EdgelistNotFoundError("Edgelist not found in pxl file")
        return lazy_data_frame

    def read_polarization(self) -> Optional[pd.DataFrame]:
        """Read the polarization data from the .pxl file."""
        return self.read_dataframe(self.POLARIZATION_KEY)

    def read_colocalization(self) -> Optional[pd.DataFrame]:
        """Read the colocalization data from the .pxl file."""
        return self.read_dataframe(self.COLOCALIZATION_KEY)

    def read_metadata(self) -> Dict:
        """Read the metadata from the .pxl file."""
        self._set_to_read_mode()
        try:
            with self._file_system.open(self.METADATA_KEY, "r") as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return {}

    def read_precomputed_layouts(
        self,
    ) -> PreComputedLayouts:
        """Read pre-computed layouts from the .pxl file."""
        layouts_lazy = self.read_dataframe_lazy(self.LAYOUTS_KEY)
        if layouts_lazy is None:
            return PreComputedLayouts.create_empty()
        return PreComputedLayouts(layouts_lazy=layouts_lazy)

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write the given metadata to the .pxl file."""
        self._set_to_write_mode()
        self._check_if_writeable(self.METADATA_KEY)
        with self._file_system.open(self.METADATA_KEY, "w") as f:
            f.write(json.dumps(metadata))

    def write_edgelist(
        self, edgelist: pd.DataFrame, partitioning: Optional[list[str]] = None
    ) -> None:
        """Write the given edgelist to the .pxl file."""
        self.write_dataframe(edgelist, self.EDGELIST_KEY, partitioning)

    def write_polarization(self, polarization: pd.DataFrame) -> None:
        """Write the given polarization data to the .pxl file."""
        self.write_dataframe(polarization, self.POLARIZATION_KEY)

    def write_colocalization(self, colocalization: pd.DataFrame) -> None:
        """Write the given colocalization data to the .pxl file."""
        self.write_dataframe(colocalization, self.COLOCALIZATION_KEY)

    def write_precomputed_layouts(
        self,
        layouts: Optional[PreComputedLayouts],
    ) -> None:
        """Write pre-computed layouts to the data store."""
        if layouts is None:
            logger.debug("No layouts to write, will skip.")
            return

        logger.debug("Starting to write layouts...")

        self._set_to_write_mode()
        self._check_if_writeable(self.LAYOUTS_KEY)

        # This is a work around for the fact that sinking into parquet files
        # from multiple sources is not supported. We therefore do this somewhat
        # round about thing of first writing the parquet files to
        # a temporary directory and then zipping them into the .pxl file.
        with TemporaryDirectory(prefix="pixelator-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            local_tmp_target = tmp_path / "local.layouts.parquet"
            layouts.write_parquet(local_tmp_target, partitioning=layouts.partitioning)

            for file_ in local_tmp_target.rglob("*"):
                if file_.is_file():
                    # TODO Make sure written without compression
                    file_name = file_.relative_to(local_tmp_target)
                    self._file_system.zip.write(
                        file_, arcname=f"{self.LAYOUTS_KEY}/{file_name}"
                    )

        logger.debug("Completed writing layouts...")

    def save(self, dataset: PixelDataset, force_overwrite: bool = False) -> None:
        """Save the given PixelDataset to the .pxl file."""
        path = Path(self.path)
        if path.exists():
            if force_overwrite:
                logger.warning("Overwriting existing .pxl file at %s", self.path)
                path.unlink()
            else:
                raise CannotOverwriteError(
                    "Cannot overwrite existing .pxl file at %s, use `force_overwrite=True` "
                    "overwrite the existing file." % self.path
                )

        self._set_to_write_mode()
        logger.debug("Writing anndata")
        self.write_anndata(dataset.adata)
        # TODO Consider using default partitioning for edgelist here
        logger.debug("Writing edgelist")
        self.write_edgelist(dataset.edgelist)
        logger.debug("Writing metadata")
        self.write_metadata(dataset.metadata)
        if dataset.polarization is not None and dataset.polarization.shape[0] > 0:
            logger.debug("Writing polarization scores")
            self.write_polarization(dataset.polarization)

        if dataset.colocalization is not None and dataset.colocalization.shape[0] > 0:
            logger.debug("Writing colocalization scores")
            self.write_colocalization(dataset.colocalization)

        if not dataset.precomputed_layouts.is_empty:
            logger.debug("Writing precomputed layouts")
            # This speeds things up massively when you have many, very small
            # layouts, like we do in some test data.
            self.write_precomputed_layouts(
                dataset.precomputed_layouts,
            )

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
        self._set_to_write_mode()

        self._check_if_writeable(key)

        with self._file_system.open(key, "wb", compression=None) as f:
            dataframe.to_csv(f, compression="gzip", index=False)

    def read_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Read a dataframe from the .pxl file."""
        self._set_to_read_mode()
        return self._read_dataframe_from_zip(key)

    def read_dataframe_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        """Read a dataframe lazily from a zip file (NB: Not implemented!)."""
        raise NotImplementedError(
            "You are trying to read data lazily from a csv-based pxl file. "
            "This is currently not supported. "
            "You can fix this issue by converting your pxl file by saving it "
            "as a parquet based pxl file."
        )

    def write_precomputed_layouts(
        self,
        layouts: Optional[PreComputedLayouts],
    ) -> None:
        """Write pre-computed layouts to the data store (NB: Not implemented!)."""
        raise NotImplementedError(
            "You are trying to write precomputed layouts to a csv-based pxl file. "
            "This is not supported. Please save your pxl file as a parquet based pxl file "
            "instead."
        )

    def _read_dataframe_from_zip(self, key: str) -> Optional[pd.DataFrame]:
        self._set_to_read_mode()
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

        Optionally provided a `partitioning` to create a hive partitioned parquet file,
        i.e. a directory structure with one level per partitioning provided, and parquet
        files as leaves.

        :param dataframe: The dataframe to write.
        :param key: The key of the dataframe to write.
        :param partitioning: The partitioning to use when writing the dataframe.
        """
        DEFAULT_COMPRESSION = "zstd"

        self._set_to_write_mode()
        if partitioning:
            file_options = ds.ParquetFileFormat().make_write_options(
                compression=DEFAULT_COMPRESSION
            )
            for _, data in dataframe.groupby(partitioning, observed=True):
                ds.write_dataset(
                    pa.Table.from_pandas(data, preserve_index=False),
                    f"{key}/",
                    filesystem=self._file_system,
                    format="parquet",
                    partitioning_flavor="hive",
                    partitioning=partitioning,
                    use_threads=False,
                    file_options=file_options,
                    existing_data_behavior="overwrite_or_ignore",
                )
            return

        self._check_if_writeable(key)
        pq.write_table(
            pa.Table.from_pandas(dataframe, preserve_index=False),
            where=key,
            filesystem=self._file_system,
            compression=DEFAULT_COMPRESSION,
            # We want all the data to go into one
            # parquet row group to allow for maximum compression
            # when we write the data here.
            # the `or 1` ensures that it does not raise if
            # the dataframe is empty.
            row_group_size=len(dataframe) or 1,
        )

    def read_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Read a dataframe from the .pxl file."""
        self._set_to_read_mode()
        return self._read_dataframe_from_zip(key)

    def read_dataframe_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        """Read a dataframe lazily from a zip file."""
        self._set_to_read_mode()
        return self._read_dataframe_from_zip_lazy(key)

    def _read_dataframe_from_zip(self, key: str) -> Optional[pd.DataFrame]:
        self._set_to_read_mode()
        df = self._read_dataframe_from_zip_lazy(key)
        if df is None:
            return None
        return df.collect().to_pandas()

    def _read_dataframe_from_zip_lazy(self, key: str) -> Optional[pl.LazyFrame]:
        try:
            self._set_to_read_mode()
            dataset = ds.dataset(
                key,
                filesystem=self._file_system,
                partitioning="hive",
                format="parquet",
                partition_base_dir=key,
            )
            return pl.scan_pyarrow_dataset(dataset)
        except FileNotFoundError:
            return None
