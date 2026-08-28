"""Core PNA pixel dataset object.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import copy
import json
import warnings
from functools import cache
from pathlib import Path
from typing import Iterable

from anndata import AnnData

from pixelator.common.graph.backends.protocol import (
    DEFAULT_LAYOUT_ALGORITHM,
    SupportedLayoutAlgorithm,
)
from pixelator.pna.pixeldataset.config import PixelDatasetConfig
from pixelator.pna.pixeldataset.edgelist import Edgelist
from pixelator.pna.pixeldataset.io import PixelDataViewer, PxlFile, Query
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.pixeldataset.layouts import Layouts
from pixelator.pna.pixeldataset.precomputed_layouts import PreComputedLayouts
from pixelator.pna.pixeldataset.proximity import Proximity
from pixelator.pna.utils import normalize_input_to_set


class PNAPixelDataset:
    """A PixelDataset is a collection of samples, components, and markers.

    This class provides a high-level interface to the data stored in one or more .pxl files.
    You can build a PixelDataset from one or more .pxl files, and then use the various methods
    to filter and access the underlying data in different ways.

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

        Args:
            view: The PixelDataViewer instance to use for accessing the data.
            config: The configuration for the dataset.
            active_components: The components to include in the dataset.
            active_markers: The markers to include in the dataset.
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
        """Alias for `from_pxl_files`.

        Args:
            pxl_files: The .pxl files to include in the dataset. Can be a list of paths or a
                dictionary with sample names as keys and paths as values.
            config: The configuration for the dataset.
        """
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

        Args:
            pxl_files: The .pxl files to include in the dataset. Can be a list of paths or a
                dictionary with sample names as keys and paths as values.
            config: The configuration for the dataset.
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

        This can be used to write custom queries to the underlying data via
        :class:`~pixelator.pna.pixeldataset.io.query_builder.Query` and
        :meth:`~pixelator.pna.pixeldataset.io.pixel_data_viewer.PixelDataViewer.open`.

        Alternatively, inside the same ``with`` block you can call
        ``session.get_connection()`` and use the DuckDB Python API directly.

        You can find more information about the duckdb API here:
        https://duckdb.org/docs/api/python/overview

        Typically you do not need to bother with using the connection
        directly, but for certain advanced use cases it can boost performance
        by quite a bit.

        .. code-block:: python

            from pixelator.pna.pixeldataset import PixelDataset
            from pixelator.pna.pixeldataset.io import Query

            pxl_files = ...
            pxl_dataset = PixelDataset.from_pxl_files(pxl_files)
            with pxl_dataset.view.open() as session:
                df = session.execute_eager(
                    Query("SELECT * FROM edgelist WHERE marker_1 = $m", {"m": "CD3"})
                ).to_pandas()


        Returns:
            The PixelDataViewer instance used by the dataset.
        """
        return self._view

    def adata(
        self,
        add_log1p_transform: bool = True,
        add_clr_transform: bool = True,
    ) -> AnnData:
        """Return the AnnData instance for the dataset.

        This will be filtered to only include the active samples, components, and markers.

        Args:
            add_log1p_transform: If True, add the log1p transformation to the data.
            add_clr_transform: If True, add the clr transformation to the data.

        Returns:
            The AnnData instance for the dataset.
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


        Returns:
            The Edgelist instance for the dataset.
        """
        return Edgelist(
            self.view,
            components=self._active_components,
            adata_helper=self._adata_helper,
        )

    def proximity(
        self,
        add_marker_counts: bool = True,
        add_logratio: bool = True,
        calculate_from_edgelist: bool = False,
    ) -> Proximity:
        """Return the Proximity instance for the dataset.

        This will be filtered to only include the active samples, components, and markers.

        Args:
            add_marker_counts: If True, add the marker counts to the proximity data.
            add_logratio: If True, add the logratio to the proximity data.
            calculate_from_edgelist: Calculate from edgelist.

        Returns:
            The Proximity instance for the dataset.
        """
        return Proximity(
            self.view,
            components=self._active_components,
            markers=self._active_markers,
            adata_helper=self._adata_helper,
            add_marker_counts=add_marker_counts,
            add_log2_ratio=add_logratio,
            calculate_from_edgelist=calculate_from_edgelist,
        )

    def precomputed_layouts(
        self, add_marker_counts: bool = True, add_spherical_norm: bool = False
    ) -> PreComputedLayouts:
        """Return the PreComputedLayouts instance for the dataset.

        .. deprecated:: Unreleased
            Use :meth:`layouts` to compute Layouts on the fly instead. This
            method still reads a stored ``layouts`` table when one exists, and
            will be removed in a future release.

        Args:
            add_marker_counts: If True, add the marker counts to the precomputed layouts.
            add_spherical_norm: If True, add spherical coordinates to dataframe This will be
                filtered to only include the active samples and components.

        Returns:
            The PreComputedLayouts instance for the dataset.
        """
        warnings.warn(
            "precomputed_layouts() is deprecated; use layouts() to compute "
            "Layouts on the fly. In the future this method will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PreComputedLayouts(
            self.view,
            components=self._active_components,
            adata_helper=self._adata_helper,
            add_marker_counts=add_marker_counts,
            add_spherical_norm=add_spherical_norm,
        )

    def layouts(
        self,
        algorithm: SupportedLayoutAlgorithm = DEFAULT_LAYOUT_ALGORITHM,
        add_marker_counts: bool = True,
        add_spherical_norm: bool = False,
        random_seed: int | None = None,
        **kwargs,
    ) -> Layouts:
        """Return Layouts for the active Components.

        A Layout places each node of a Component graph in 2D or 3D so
        you can visualize the cell and for example color nodes by marker.
        Computation runs when you materialize the result (for example
        :meth:`~pixelator.pna.pixeldataset.layouts.Layouts.to_df`).
        This replaces the deprecated :meth:`precomputed_layouts`.

        Choose a Layout algorithm with ``algorithm``. The default
        ``coarsened_pmds_3d`` is the usual choice for PNA: it is fast on large
        Components and produces a 3D Layout suitable for plotting.

        Available Layout algorithms:

        - ``coarsened_pmds_3d`` (default): 3D layout algorithm that uses a
          pre-coarsening step. Fast and robust to structural artifacts.
        - ``wpmds_3d``: 3D weighted PMDS; a good alternative when you want a
          full (non-coarsened) PMDS Layout.
        - ``pmds`` / ``pmds_3d``: 2D or 3D PMDS without coarsening.
        - ``spectral_3d``: 3D spectral Layout. Extremely fast, generates high
          quality layouts but can be sensitive to structural artifacts.
        - ``fruchterman_reingold`` / ``fruchterman_reingold_3d``: force-directed;
          slower on large Components.
        - ``kamada_kawai`` / ``kamada_kawai_3d``: force-directed; slower on
          large Components.

        For most cases prefer ``coarsened_pmds_3d``, ``wpmds_3d``, or ``pmds`` (in that order).
        On PNA data they are faster and produce better results than the
        force-directed algorithms.

        .. code-block:: python

                df = pxl_dataset.filter(components=component_id).layouts().to_df()

        Args:
            algorithm: Layout algorithm to use. Defaults to
                ``DEFAULT_LAYOUT_ALGORITHM`` (``coarsened_pmds_3d``).
            add_marker_counts: If True, add per-node marker counts.
            add_spherical_norm: If True, add spherical unit-vector columns.
            random_seed: Seed for Layout algorithms with a stochastic element.
            **kwargs: Forwarded to the underlying Layout algorithm.

        Returns:
            A Layouts collection for the active Components.
        """
        return Layouts(
            self.view,
            components=self._active_components,
            adata_helper=self._adata_helper,
            algorithm=algorithm,
            add_marker_counts=add_marker_counts,
            add_spherical_norm=add_spherical_norm,
            random_seed=random_seed,
            algorithm_kwargs=kwargs,
        )

    def metadata(
        self,
    ) -> dict:
        """Return the metadata for the dataset."""
        with self.view.open() as session:
            metadata_df = session.execute_eager(Query("SELECT * FROM metadata", {}))
            maybe_metadata = [json.loads(x[0]) for x in metadata_df.iter_rows()]
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

        Filtering by components will apply to all data modalities (i.e. adata, edgelist, proximity,
        layouts, and precomputed layouts).
        However, filtering by markers will only apply to the adata and proximity data modalities,
        since filtering
        by markers in the edgelist and precomputed layouts will cause components to break up.

        Note that filtering is done lazily, so creating new filters is cheap. The actual filtering
        will only be done
        once the underlying data is accessed.

        Args:
            samples: (The samples to include in the dataset (default): None means no filter is
                applied).
            components: (The components to include in the dataset (default): None means no filter is
                applied).
            markers: (The markers to include in the dataset (default): None means no filter is
                applied).

        Returns:
            A new PixelDataset with the specified samples, components, and markers
        Raises:
            ValueError: if all of the specified samples, components, or markers do not exist in the
                dataset.
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
