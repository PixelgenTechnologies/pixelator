"""Module for PixelDataset backends.

Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations

from functools import cached_property
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
)

import pandas as pd
import polars as pl
from anndata import AnnData

from pixelator.pixeldataset.file_formats import PixelFileFormatSpec
from pixelator.pixeldataset.utils import (
    _enforce_edgelist_types,
)
from pixelator.types import PathType


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
    def edgelist_lazy(self) -> pl.LazyFrame:
        """Return the edgelist as a LazyFrame."""

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
        copy: bool = True,
        allow_edgelist_to_be_empty: bool = False,
    ) -> None:
        """Create a new instance of ObjectBasedPixelDatasetBackend.

        :param adata: an AnnData instance
        :param edgelist: an edgelist dataframe
        :param metadata: a dict with metadata, defaults to None
        :param polarization: a polarization dataframe, defaults to None
        :param colocalization: a colocalization dataframe, defaults to None
        :param copy: decide if the input data should be copied or not. Defaults to True.
        :param allow_edgelist_to_be_empty: allow the edgelist to be empty.
                                           Defaults to False.
        :raises AssertionError: if `adata` or `edgelist` contains no data.
        """
        if adata is None or adata.n_obs == 0:
            raise AssertionError("adata cannot be empty")

        if edgelist is None or edgelist.shape[0] == 0:
            if not allow_edgelist_to_be_empty:
                raise AssertionError("edgelist cannot be empty")
            edgelist = _enforce_edgelist_types(pd.DataFrame())

        self._edgelist = _enforce_edgelist_types(edgelist.copy() if copy else edgelist)
        self._adata = adata.copy() if copy else adata
        self._metadata = metadata
        self._polarization = None
        if polarization is not None:
            self._polarization = polarization.copy() if copy else polarization
        self._colocalization = None
        if colocalization is not None:
            self._colocalization = colocalization.copy() if copy else colocalization

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
    def edgelist_lazy(self) -> pl.LazyFrame:
        """Get a lazy frame representation of the edgelist."""
        return pl.LazyFrame(self._edgelist)

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

    @property
    def edgelist_lazy(self) -> Optional[pl.LazyFrame]:
        """Get a lazy frame representation of the edgelist."""
        return self._file_format.deserialize_dataframe_lazy(
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
