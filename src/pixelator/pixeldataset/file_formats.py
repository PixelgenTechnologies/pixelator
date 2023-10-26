"""Module for PixelDataset file formats.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations

import gzip
import json
import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Type,
)
from zipfile import ZIP_STORED, ZipFile

import pandas as pd
import polars as pl
import pyarrow.dataset as ds
from anndata import AnnData
from fsspec.implementations.zip import ZipFileSystem

if TYPE_CHECKING:
    from pixelator.pixeldataset import PixelDataset
from pixelator.pixeldataset.utils import (
    read_anndata,
    write_anndata,
)
from pixelator.types import PathType

logger = logging.getLogger(__name__)


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

    @staticmethod
    @abstractmethod
    def deserialize_dataframe_lazy(path: PathType, key: str) -> Optional[pl.LazyFrame]:
        """Deserialize a data frame from a .pxl file as a lazy frame.

        Deserialize a data frame from a .pxl file at `path`, which has been
        stored with the file name `key` in the zip archive, as a lazy frame.

        :param path: the path to a pxl file
        :param key: the file name that the data frame was stored under
        :return: a `pl.LazyFrame` instance or None
        :rtype: Optional[pl.LazyFrame]
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

            if dataset.polarization is not None and dataset.polarization.shape[0] > 0:
                # create and save temporary polarization scores
                file = tempfile.mkstemp(suffix=".csv.gz")[1]
                self.serialize_dataframe(dataset.polarization, file)
                zip_archive.write(file, self.POLARIZATION_KEY)
                Path(file).unlink()

            if (
                dataset.colocalization is not None
                and dataset.colocalization.shape[0] > 0
            ):
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
    def deserialize_dataframe_lazy(path: PathType, key: str) -> Optional[pl.LazyFrame]:
        """Deserialize a lazy frame from the give path."""
        return PixelFileCSVFormatSpec._read_dataframe_from_zip_lazy(path, key)

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

    @staticmethod
    def _read_dataframe_from_zip_lazy(
        path: PathType, key: str
    ) -> Optional[pl.LazyFrame]:
        raise NotImplementedError(
            "You are trying to read data lazily from a csv-based pxl file. "
            "This is currently not supported. "
            "You can fix this issue by converting your pxl file by saving it "
            "as a parquet based pxl file."
        )


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
        dataframe.to_parquet(
            path, engine="fastparquet", compression="zstd", index=False
        )

    @staticmethod
    def deserialize_dataframe(path: PathType, key: str) -> pd.DataFrame:
        """Deserialize a dataframe from the give path."""
        return PixelFileParquetFormatSpec._read_dataframe_from_zip(path, key)

    @staticmethod
    def deserialize_dataframe_lazy(path: PathType, key: str) -> Optional[pl.LazyFrame]:
        """Deserialize a dataframe from the give path."""
        return PixelFileParquetFormatSpec._read_dataframe_from_zip_lazy(path, key)

    @staticmethod
    def _read_dataframe_from_zip(path: PathType, key: str) -> Optional[pd.DataFrame]:
        with ZipFile(path, "r") as zip_archive:
            members = zip_archive.namelist()
            if key not in members:
                return None
            with zip_archive.open(key) as f:
                try:
                    return pd.read_parquet(f, engine="fastparquet")
                except ValueError:
                    return pl.read_parquet(f).to_pandas()  # type: ignore

    @staticmethod
    def _read_dataframe_from_zip_lazy(
        path: PathType, key: str
    ) -> Optional[pl.LazyFrame]:
        file_system = ZipFileSystem(path)
        dataset = ds.dataset(key, filesystem=file_system)
        return pl.scan_pyarrow_dataset(dataset)
