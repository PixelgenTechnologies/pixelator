"""Core PNA pixel dataset object.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Iterable

from anndata import AnnData

from pixelator.pna.pixeldataset.config import PixelDatasetConfig
from pixelator.pna.pixeldataset.edgelist import Edgelist
from pixelator.pna.pixeldataset.io import PixelDataViewer, PxlFile
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pna.pixeldataset.proximity import Proximity
from pixelator.pna.utils import normalize_input_to_set


class PNAPixelDataset:
    """A PixelDataset is a collection of samples, components, and markers.

    This class provides a high-level interface to the data stored in one or more .pxl files.
    You can build a PixelDataset from one or more .pxl files, and then use the various methods
    to filer and access the underlying data in different ways.

    .. code-block:: python
        from pathlib import Path
        from pixelator.pna.pixeldataset import PixelDataset

        pxl_files = Path("<dir with pxl files>").glob("*.pxl")
        pxl_dataset = PixelDataset.from_pxl_files(pxl_files)

    To filter data you can do:
    .. code-block:: python
        ten_components = pxl_dataset.adata.obs.index[:10]
        pxl_dataset.filter(components=ten_components)
    """

    def __init__(
        self,
        view: PixelDataViewer,
        config: PixelDatasetConfig | None = None,
        active_components: Iterable[str] | str | None = None,
        active_markers: Iterable[str] | str | None = None,
    ):
        """Create a new PixelDataset instance.

        Note that setting any of the `active_*` parameters to None will include
        all samples, components, or markers.

        :param view: The PixelDataViewer instance to use for accessing the data.
        :param config: The configuration for the dataset.
        :param active_components: The components to include in the dataset.
        :param active_markers: The markers to include in the dataset.
        """
        self._view = view
        if config is None:
            config = PixelDatasetConfig()
        self._config = config

        self._active_components = normalize_input_to_set(active_components)
        self._active_markers = normalize_input_to_set(active_markers)
        self._adata_helper = AnnDataHelper(
            view=self._view,
            components=self._active_components,
            markers=self._active_markers,
            adata_join_strategy=self._config.adata_join_method,
        )

    @staticmethod
    def from_files(
        pxl_files: Path
        | Iterable[Path]
        | Iterable[PxlFile]
        | PxlFile
        | dict[str, Path],
        config: PixelDatasetConfig | None = None,
    ) -> PNAPixelDataset:
        """Alias for `from_pxl_files`."""
        return PNAPixelDataset.from_pxl_files(pxl_files, config)

    @staticmethod
    def from_pxl_files(
        pxl_files: Path
        | Iterable[Path]
        | Iterable[PxlFile]
        | PxlFile
        | dict[str, Path],
        config: PixelDatasetConfig | None = None,
    ) -> PNAPixelDataset:
        """Create a new PixelDataset from one or more .pxl files.

        If you pass a list of .pxl files the name of the samples
        will be inferred from the sample name in the file.

        If you pass a dictionary of .pxl files the keys will be used as the sample names.

        :param pxl_files: The .pxl files to include in the dataset.
                          Can be a list of paths or a dictionary with sample names
                          as keys and paths as values.
        :param config: The configuration for the dataset.
        """
        if isinstance(pxl_files, Path):
            return PNAPixelDataset(
                PixelDataViewer.from_files(pxl_files=[PxlFile(pxl_files)]),
                config=config,
            )

        if isinstance(pxl_files, dict):
            return PNAPixelDataset(
                PixelDataViewer.from_sample_to_file_mappings(
                    {k: PxlFile(v) for k, v in pxl_files.items()}
                ),
                config=config,
            )

        if isinstance(pxl_files, PxlFile):
            return PNAPixelDataset(
                PixelDataViewer.from_files(pxl_files=[pxl_files]),
                config=config,
            )

        pxl_files = list(pxl_files)  # type: ignore
        # Either you are a PxlFile, then let's go!
        if all(isinstance(f, PxlFile) for f in pxl_files):
            return PNAPixelDataset(
                view=PixelDataViewer.from_files(pxl_files),  # type: ignore
                config=config,
            )

        # Or we will assume you are path and try that!
        return PNAPixelDataset(
            view=PixelDataViewer.from_files(
                pxl_files=[PxlFile(f) for f in pxl_files]  # type: ignore
            ),
            config=config,
        )

    def sample_names(self) -> set[str]:
        """Return the set of sample names in the project."""
        return set(self.adata().obs["sample"].unique().tolist())

    def components(self) -> set[str]:
        """Return the set of component names in the project."""
        return set(self.adata().obs.index.to_list())

    def markers(self) -> set[str]:
        """Return the set of marker names in the project."""
        return set(self.adata().var.index.to_list())

    @property
    def view(self) -> PixelDataViewer:
        """Return the PixelDataViewer instance used by the dataset.

        This can be used to write custom queries to the underlying data, using
        the duckdb connection API.

        You can find more information about the duckdb API here:
        https://duckdb.org/docs/api/python/overview

        Typically you do not need to bother with using the connection
        directly, but for certain advanced use cases it can boost performance
        by quite a bit.

        .. code-block:: python
            from pixelator.pna.pixeldataset import PixelDataset

            pxl_files = ...
            pxl_dataset = PixelDataset.from_pxl_files(pxl_files)
            with pxl_dataset.view as connection:
                df = connection.sql("SELECT * FROM edgelist WHERE markers = 'CD3'").to_df()

        :return: The PixelDataViewer instance used by the dataset.
        """
        return self._view

    def adata(
        self,
        add_log1p_transform: bool = True,
        add_clr_transform: bool = True,
    ) -> AnnData:
        """Return the AnnData instance for the dataset.

        This will be filtered to only include the active samples, components, and markers.
        :param add_log1p_transform: If True, add the log1p transformation to the data.
        :param add_clr_transform: If True, add the clr transformation to the data.
        :return: The AnnData instance for the dataset.
        """
        return self._adata_helper.read_adata(
            add_log1p_transform=add_log1p_transform,
            add_clr_transform=add_clr_transform,
        )

    def edgelist(
        self,
    ) -> Edgelist:
        """Return the Edgelist instance for the dataset.

        This will be filtered to only include the active samples and components.
        :return: The Edgelist instance for the dataset.
        """
        return Edgelist(self.view, components=self._active_components)

    def proximity(
        self,
        add_marker_counts: bool = True,
        add_logratio: bool = True,
    ) -> Proximity:
        """Return the Proximity instance for the dataset.

        This will be filtered to only include the active samples, components, and markers.

        :param add_marker_counts: If True, add the marker counts to the proximity data.
        :param add_logratio: If True, add the logratio to the proximity data.
        :return: The Proximity instance for the dataset.
        """
        return Proximity(
            self.view,
            components=self._active_components,
            markers=self._active_markers,
            adata_helper=self._adata_helper,
            add_marker_counts=add_marker_counts,
            add_log2_ratio=add_logratio,
        )

    def precomputed_layouts(
        self, add_marker_counts: bool = True, add_spherical_norm: bool = False
    ) -> PreComputedLayouts:
        """Return the PreComputedLayouts instance for the dataset.

        :param add_marker_counts: If True, add the marker counts to the precomputed layouts.
        :param add_spherical_norm: If True, add spherical coordinates to dataframe
        This will be filtered to only include the active samples and components.
        :return: The PreComputedLayouts instance for the dataset.
        """
        return PreComputedLayouts(
            self.view,
            components=self._active_components,
            add_marker_counts=add_marker_counts,
            add_spherical_norm=add_spherical_norm,
        )

    def metadata(
        self,
    ) -> dict:
        """Return the metadata for the dataset."""
        with self.view as connection:
            maybe_metadata = [
                json.loads(x[0])
                for x in connection.sql("SELECT * FROM metadata").fetchall()
            ]
            if not maybe_metadata:
                return {}

            metadata: dict = {}
            for metadata_dict in maybe_metadata:
                metadata[metadata_dict["sample_name"]] = metadata_dict
            return metadata

    @staticmethod
    def _copy_or_none(values_or_none):
        if values_or_none is None:
            return None
        return values_or_none.copy()

    def filter(
        self,
        samples: Iterable[str] | str | None = None,
        components: Iterable[str] | str | None = None,
        markers: Iterable[str] | str | None = None,
    ) -> PNAPixelDataset:
        """Filter the dataset to only include the specified samples, components, and markers.

        Filtering by components will apply to all data modalities (i.e. adata, edgelist, proximity, and precomputed layouts).
        However, filtering by markers will only apply to the adata and proximity data modalities, since filtering
        by markers in the edgelist and precomputed layouts will cause components to break up.

        Note that filtering is done lazily, so creating new filters is cheap. The actual filtering will only be done
        once the underlying data is accessed.

        :param samples: The samples to include in the dataset (default: None means no filter is applied).
        :param components: The components to include in the dataset (default: None means no filter is applied).
        :param markers: The markers to include in the dataset (default: None means no filter is applied).
        :raises ValueError: if all of the specified samples, components, or markers do not exist in the dataset.
        :return: A new PixelDataset with the specified samples, components, and markers
        """
        samples = normalize_input_to_set(samples)
        components = normalize_input_to_set(components)
        markers = normalize_input_to_set(markers)

        errors = []
        if samples and not samples.issubset(self.sample_names()):
            errors.append(
                "One or more of the specified samples do not exist in the dataset."
            )

        if components and not components.issubset(self.components()):
            errors.append(
                "One or more of the specified components do not exist in the dataset."
            )

        if markers and not markers.issubset(self.markers()):
            errors.append(
                "One or more of the specified markers do not exist in the dataset."
            )

        if errors:
            message = ["Failed to filter, for the following reasons: "]
            message.extend(errors)
            raise ValueError("\n".join(message))

        active_components = components or self._copy_or_none(self._active_components)
        active_markers = markers or self._copy_or_none(self._active_markers)
        new_view = self._view.filter_samples(samples) if samples else self._view

        return PNAPixelDataset(
            view=new_view,
            config=copy.copy(self._config),
            active_components=active_components,
            active_markers=active_markers,
        )

    def __repr__(self) -> str:
        """Return a string representation of the PixelDataset."""
        return str(self)

    def __str__(self) -> str:
        """Return a string representation of the PixelDataset."""
        return f"""PixelatorProject(with {len(self.sample_names())} samples)"""

    def _ipython_display_(self):
        """Display the PixelDataset in Jupyter notebooks."""
        return print(self.describe())

    def describe(self) -> str:
        """Return a description of the PixelDataset."""
        description = [f"""PixelDataset with {len(self.sample_names())} samples"""]
        description.append("")
        description += ["""Mapping the following samples to files:"""]
        for sample_name, file in self.view.sample_to_file_mappings.items():
            description.append(f"\tSample: {sample_name}, File: {file}")

        description.append("")
        adata = self.adata()
        description += ["""In total it contains:"""]
        description.append(f"{len(adata.obs)} components, {len(adata.var)} markers")
        return "\n".join(description)
