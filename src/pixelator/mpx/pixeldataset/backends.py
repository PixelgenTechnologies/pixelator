"""Module for PixelDataset backends.

Copyright © 2023 Pixelgen Technologies AB.
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

from pixelator.common.types import PathType
from pixelator.mpx.pixeldataset.datastores import PixelDataStore
from pixelator.mpx.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.mpx.pixeldataset.utils import (
    _enforce_edgelist_types,
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

    @property
    def precomputed_layouts(self) -> PreComputedLayouts:
        """Get the precomputed layouts for the component graphs.

        Please note that since these have been pre-computed, if you have made
        changes to the underlying data, the layout might not be valid anymore.
        """
        ...

    @precomputed_layouts.setter
    def precomputed_layouts(self, value: PreComputedLayouts | None) -> None:
        """Set the precomputed layouts for the component graphs."""
        ...


class ObjectBasedPixelDatasetBackend(PixelDatasetBackend):
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
        precomputed_layouts: Optional[PreComputedLayouts] = None,
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

        self._precomputed_layouts = PreComputedLayouts.create_empty()
        if precomputed_layouts is not None:
            self._precomputed_layouts = precomputed_layouts

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

    @property
    def precomputed_layouts(self) -> PreComputedLayouts:
        """Get the precomputed layouts."""
        if self._precomputed_layouts is None:
            return PreComputedLayouts.create_empty()
        return self._precomputed_layouts

    @precomputed_layouts.setter
    def precomputed_layouts(self, value: PreComputedLayouts | None) -> None:
        """Set the precomputed layouts."""
        if value is None:
            self._precomputed_layouts = PreComputedLayouts.create_empty()
        self._precomputed_layouts = value


class FileBasedPixelDatasetBackend(PixelDatasetBackend):
    """A file based backend for PixelDataset.

    `FileBasedPixelDatasetBackend` is used to lazily fetch information from
    a .pxl file. Once the file has been read the data will be cached
    in memory.
    """

    def __init__(
        self, path: PathType, datastore: Optional[PixelDataStore] = None
    ) -> None:
        """Create a filebased backend instance.

        Create a backend, fetching information from the .pxl file
        in `path`.

        :param path: Path to the .pxl file
        """
        self._path = path
        if not datastore:
            datastore = PixelDataStore.guess_datastore_from_path(path)
        self._datastore = datastore
        self._precomputed_layouts: PreComputedLayouts | None = None

    @cached_property
    def adata(self) -> AnnData:
        """Get the AnnData object for the pixel dataset."""
        return self._datastore.read_anndata()

    @cached_property
    def edgelist(self) -> pd.DataFrame:
        """Get the edge list object for the pixel dataset."""
        return self._datastore.read_edgelist()

    @property
    def edgelist_lazy(self) -> pl.LazyFrame:
        """Get a lazy frame representation of the edgelist."""
        return self._datastore.read_edgelist_lazy()

    @cached_property
    def polarization(self) -> Optional[pd.DataFrame]:
        """Get the polarization object for the pixel dataset."""
        return self._datastore.read_polarization()

    @cached_property
    def colocalization(self) -> Optional[pd.DataFrame]:
        """Get the colocalization object for the pixel dataset."""
        return self._datastore.read_colocalization()

    @cached_property
    def metadata(self) -> Optional[Dict]:
        """Get the metadata object for the pixel dataset."""
        return self._datastore.read_metadata()

    @property
    def precomputed_layouts(self) -> PreComputedLayouts:
        """Get the precomputed layouts."""
        # If it is None it means it is uninitialized, and we should
        # attempt to read it lazily

        if isinstance(self._precomputed_layouts, PreComputedLayouts):
            return self._precomputed_layouts

        if self._precomputed_layouts is None:
            self._precomputed_layouts = self._datastore.read_precomputed_layouts()
        return self._precomputed_layouts

    @precomputed_layouts.setter
    def precomputed_layouts(self, value: PreComputedLayouts | None) -> None:
        """Set the precomputed layouts."""
        if value is None:
            self._precomputed_layouts = PreComputedLayouts.create_empty()
            return
        self._precomputed_layouts = value
