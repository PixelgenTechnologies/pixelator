"""PNA Pixel Dataset.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import copy
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from anndata import AnnData, ImplicitModificationWarning

from pixelator.common.statistics import clr_transformation, log1p_transformation
from pixelator.pna.graph import PNAGraph, PNAGraphBackend
from pixelator.pna.pixeldataset.io import (
    InplacePixelDataFilterer,
    PixelDataQuerier,
    PixelDataViewer,
    PxlFile,
    copy_databases,
)
from pixelator.pna.utils import normalize_input_to_list, normalize_input_to_set


def read(paths: Path | list[Path] | str | list[str]) -> PNAPixelDataset:
    """Read a PNAPixelDataset from one or more provided .pxl file(s).

    :param path: path to the file to read
    :return: an instance of `PNAPixelDataset`
    """
    if not isinstance(paths, list):
        paths = [paths]  # type: ignore
    normalized_paths = [Path(p) for p in paths]  # type: ignore
    return PNAPixelDataset.from_pxl_files(normalized_paths)


@dataclass(slots=True, frozen=True, repr=True)
class Component:
    """A dataclass to hold a component and its associated graph."""

    component_id: str
    frame: pl.LazyFrame

    @property
    def graph(self) -> PNAGraph:
        """Get the graph."""
        return PNAGraph(PNAGraphBackend.from_edgelist(self.frame))


class Edgelist:
    """Representation of an edgelist.

    To get the edgelist as a pandas DataFrame, use the `to_df()` method.
    To get the edgelist as a polars DataFrame, use the `to_polars()` method.
    To access the components one at a time use the `iterator()` method.

    For memory efficient access to the edgelist, use the `to_record_batches()` method.
    This allows you to iterate over the edgelist using py.ArrowRecordBatch's.
    This is useful when you have a large edgelist that does not fit into memory,
    and your algorithm can be implemented in a streaming fashion.
    """

    def __init__(
        self,
        querier: PixelDataQuerier,
        components: str | Iterable[str] | None = None,
    ):
        """Create a new instance of Edgelist."""
        self._querier = querier
        self._components = normalize_input_to_set(components)

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        return self._components or set(self._querier.read_all_component_names())

    def _handle_backwards_compatibility(
        self, df: pl.DataFrame | pd.DataFrame
    ) -> pl.DataFrame | pd.DataFrame:
        # Handle legacy marker names
        if isinstance(df, pl.DataFrame):
            return df.rename(
                {"marker1": "marker_1", "marker2": "marker_2"}, strict=False
            )
        else:
            return df.rename(columns={"marker1": "marker_1", "marker2": "marker_2"})

    def __len__(self) -> int:
        """Get the number of edges in the edgelist."""
        return self._querier.read_edgelist_len(components=self._components)

    def is_empty(self) -> bool:
        """Check if the edgelist is empty."""
        return len(self) == 0

    def to_df(self) -> pd.DataFrame:
        """Get the edgelist as a pandas DataFrame."""
        df = self._querier.read_edgelist(components=self.components, as_pandas=True)
        return self._handle_backwards_compatibility(df)

    def to_polars(self) -> pl.DataFrame:
        """Get the edgelist as a polars DataFrame."""
        # TODO change this once we can get filtering pushdown in duckdb
        # But for now this, somewhat counter-intuitively is the faster
        df = pl.concat([df for _, df in self._iterator()])
        return self._handle_backwards_compatibility(df)

    def to_record_batches(
        self, batch_size: int = 1_000_000
    ) -> Iterable[pa.RecordBatch]:
        """Get the edgelist as a stream of pyarrow RecordBatches."""
        return self._querier.read_edgelist_stream(
            components=self.components, batch_size=batch_size
        )

    def _iterator(self) -> Iterable[tuple[str, pl.DataFrame]]:
        for component in self.components:
            yield component, self._querier.read_edgelist(components=component)

    def iterator(self) -> Iterable[Component]:
        """Get a stream of components and their graphs.

        :return: A stream of component names and associated graphs
        """
        for name, df in self._iterator():
            yield Component(
                component_id=name, frame=self._handle_backwards_compatibility(df).lazy()
            )

    def __str__(self) -> str:
        """Get a string representation of the Edgelist."""
        n_edges = len(self)
        n_components = len(self.components)
        if n_components <= 5:
            return f"EdgeList({n_edges:,} edges in component set: {self.components})"
        return f"EdgeList({n_edges:,} edges in {n_components} components)"

    def __repr__(self):
        """Get a string representation of the Edgelist."""
        return str(self)

    def _ipython_display_(self):
        """Display the Edgelist in Jupyter notebooks."""
        return print(str(self))


class Proximity:
    """Representation of a proximity data.

    This class allows you to access the proximity data as a pandas or polars
    DataFrame. From there you can analyze the data further using your favorite
    data analysis tools.
    """

    def __init__(
        self,
        querier: PixelDataQuerier,
        components: str | list[str] | set[str] | None = None,
        markers: str | list[str] | set[str] | None = None,
        add_marker_counts: bool = True,
        add_log2_ratio: bool = True,
    ):
        """Create a new instance of Proximity."""
        self._querier = querier
        self._components = normalize_input_to_set(components)
        self._markers = normalize_input_to_set(markers)
        self._add_marker_counts = add_marker_counts
        self._add_log2_ratio_col = add_log2_ratio

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        return (
            self._components
            if self._components is not None
            else set(self._querier.read_all_component_names())
        )

    @property
    def markers(self) -> set[str]:
        """Get the marker names."""
        return (
            self._markers
            if self._markers is not None
            else set(self._querier.read_all_marker_names())
        )

    def __len__(self) -> int:
        """Get the number of proximity scores."""
        return self._querier.read_proximity_len(
            components=self._components, markers=self._markers
        )

    def is_empty(self):
        """Check if the proximity data is empty."""
        return len(self) == 0

    def _add_marker_counts_to_proximity_df(
        self, adata: AnnData, proximity_df: pl.DataFrame
    ):
        marker_counts = pl.DataFrame(adata.to_df().reset_index())
        node_counts = (
            marker_counts.drop("component")
            .transpose(column_names=marker_counts["component"])
            .sum()
            .transpose(column_names=["node_counts"])
            .with_columns(component=marker_counts["component"])
        )
        marker_counts = marker_counts.unpivot(
            index="component", variable_name="marker", value_name="marker_1_count"
        )
        marker_counts = marker_counts.with_columns(
            pl.col("marker_1_count").alias("marker_2_count")
        )
        # Cast to uint32 integers to lower memory usage
        marker_counts = marker_counts.with_columns(
            marker_1_count=pl.col("marker_1_count").cast(pl.UInt32),
            marker_2_count=pl.col("marker_2_count").cast(pl.UInt32),
        )
        marker_counts = marker_counts.join(
            node_counts,
            on="component",
            how="left",
        )
        marker_counts = marker_counts.with_columns(
            marker_1_freq=pl.col("marker_1_count") / pl.col("node_counts"),
            marker_2_freq=pl.col("marker_2_count") / pl.col("node_counts"),
        )
        proximity_df = (
            proximity_df.join(
                marker_counts.select(
                    ["component", "marker", "marker_1_count", "marker_1_freq"]
                ),
                left_on=["component", "marker_1"],
                right_on=["component", "marker"],
                how="left",
            )
            .join(
                marker_counts.select(
                    ["component", "marker", "marker_2_count", "marker_2_freq"]
                ),
                left_on=["component", "marker_2"],
                right_on=["component", "marker"],
                how="left",
            )
            .with_columns(
                min_count=pl.min_horizontal("marker_1_count", "marker_2_count")
            )
        )

        return proximity_df

    def _add_log2_ratio(self, proximity):
        return proximity.with_columns(
            log2_ratio=np.log2(
                np.maximum(pl.col("join_count"), 1)
                / np.maximum(pl.col("join_count_expected_mean"), 1)
            )  # setting values <1 to 1 to avoid division by zero
        )

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._add_marker_counts:
            adata = self._querier.read_adata()
            df = self._add_marker_counts_to_proximity_df(adata, df)

        if self._add_log2_ratio_col:
            df = self._add_log2_ratio(df)

        return df

    def to_df(self) -> pd.DataFrame:
        """Get the edgelist as a pandas DataFrame."""
        return self.to_polars().to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Get the edgelist as a polars DataFrame."""
        return self._post_process(
            self._querier.read_proximity(
                components=self._components, markers=self._markers
            )
        )

    def __str__(self) -> str:
        """Get a string representation of the Proximity."""
        n_proximity = len(self)
        return ", ".join(
            [
                f"Proximity({n_proximity:,} elements",
                f"add_marker_counts={self._add_marker_counts}",
                f"add_logratio={self._add_log2_ratio_col})",
            ]
        )

    def __repr__(self):
        """Get a string representation of the Proximity."""
        return str(self)

    def _ipython_display_(self):
        return print(str(self))


class PreComputedLayouts:
    """Representation of precomputed layouts.

    This contains precomputed layouts for one or more components.
    """

    def __init__(
        self,
        querier: PixelDataQuerier,
        components: str | Iterable[str] | None = None,
        add_marker_counts: bool = True,
        add_spherical_norm: bool = False,
    ):
        """Create a new instance of PreComputedLayouts."""
        self._querier = querier
        self._components = normalize_input_to_set(components)
        self._add_marker_counts = add_marker_counts
        self._add_spherical_norm = add_spherical_norm

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        return (
            self._components
            if self._components is not None
            else set(self._querier.read_all_component_names())
        )

    def __len__(self) -> int:
        """Get the nodes in the layouts."""
        return self._querier.read_layouts_len(components=self._components)

    def is_empty(self) -> bool:
        """Check if the precomputed layouts are empty."""
        return len(self) == 0

    def to_df(self) -> pd.DataFrame:
        """Get the precomputed layouts as a pandas DataFrame."""
        return self.to_polars().to_pandas()

    def _handle_backwards_comp(self, df: pl.DataFrame) -> pl.DataFrame:
        # If the norm data is in the output (which it is for some legacy files)
        # drop them.
        return df.drop(["x_norm", "y_norm", "z_norm"], strict=False)

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._handle_backwards_comp(df)

        if self._add_spherical_norm:
            coordinates = df.select(["x", "y", "z"]).to_numpy()
            normalized_coordinates = pl.DataFrame(
                coordinates / (1 * np.linalg.norm(coordinates, axis=1))[:, None],
                schema={
                    "x_norm": pl.Float32,
                    "y_norm": pl.Float32,
                    "z_norm": pl.Float32,
                },
            )
            df = df.hstack(normalized_coordinates)

        return df

    def to_polars(self) -> pl.DataFrame:
        """Get the precomputed layouts as a polars DataFrame."""
        return self._post_process(
            self._querier.read_layouts(
                components=self._components, add_marker_counts=self._add_marker_counts
            )
        )

    def iterator(
        self, return_polars_df: bool = False
    ) -> Iterable[tuple[str, pd.DataFrame | pl.DataFrame]]:
        """Get a stream of layouts.

        Control the trade-off between memory usage and performance by setting
        the `batch_size` parameter. If you set it to a large number, you will
        use more memory, but the performance should overall be better.

        :param return_polars_df: If True, return polars DataFrames, otherwise return pandas DataFrames

        :return: A stream of layouts names and associated layout dataframes
        """
        for component in self.components:
            component_df = self._post_process(
                self._querier.read_layouts(
                    components=component, add_marker_counts=self._add_marker_counts
                )
            )
            if return_polars_df:
                yield (component, component_df)
            else:
                yield (component, component_df.to_pandas())

    def describe(self) -> str:
        """Return a description of the PreComputedLayouts."""
        return f"PreComputedLayouts({len(self.components):,} components, {len(self):,} datapoints)"

    def __str__(self):
        """Get a string representation of the PreComputedLayouts."""
        return f"PreComputedLayouts({len(self.components):,} components)"

    def __repr__(self):
        """Get a string representation of the PreComputedLayouts."""
        return str(self)

    def _ipython_display_(self):
        return print(self.describe())


@dataclass
class PixelDatasetConfig:
    """Configuration for a PixelDataset."""

    adata_join_method: Literal["inner", "outer"] = "inner"


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
        active_samples: Iterable[str] | str | None = None,
        active_components: Iterable[str] | str | None = None,
        active_markers: Iterable[str] | str | None = None,
    ):
        """Create a new PixelDataset instance.

        Note that setting any of the `active_*` parameters to None will include
        all samples, components, or markers.

        :param view: The PixelDataViewer instance to use for accessing the data.
        :param config: The configuration for the dataset.
        :param active_samples: The samples to include in the dataset.
        :param active_components: The components to include in the dataset.
        :param active_markers: The markers to include in the dataset.
        """
        self._view = view
        if config is None:
            config = PixelDatasetConfig()
        self._config = config

        self._active_samples = (
            normalize_input_to_set(active_samples)
            if active_samples
            else set(self._view.sample_names())
        )
        self._active_components = normalize_input_to_set(active_components)
        self._active_markers = normalize_input_to_set(active_markers)

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
            view=PixelDataViewer.from_files(pxl_files=[PxlFile(f) for f in pxl_files]),  # type: ignore
            config=config,
        )

    def sample_names(self) -> set[str]:
        """Return the set of sample names in the project."""
        return set(self._active_samples)  # type: ignore

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
        # TODO For performance reasons we might want to push down
        # the component and marker filtering to the individual datasets
        # but for now this should probably be fine
        adata = self._view.read_adata()
        if self._active_components:
            adata = adata[normalize_input_to_list(self._active_components), :]
        if self._active_markers:
            adata = adata[:, normalize_input_to_list(self._active_markers)]

        # add normalization layers if requested
        if [add_clr_transform, add_log1p_transform].count(True) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
                counts_df = adata.to_df()
                if add_clr_transform:
                    counts_df_clr = clr_transformation(df=counts_df, axis=1)
                    counts_df_clr.index = counts_df_clr.index.astype(str)
                    adata.obsm["clr"] = counts_df_clr

                if add_log1p_transform:
                    counts_df_log1p = log1p_transformation(df=counts_df)
                    counts_df_log1p.index = counts_df_log1p.index.astype(str)
                    adata.obsm["log1p"] = counts_df_log1p

        return adata.copy()

    def edgelist(
        self,
    ) -> Edgelist:
        """Return the Edgelist instance for the dataset.

        This will be filtered to only include the active samples and components.
        :return: The Edgelist instance for the dataset.
        """
        return Edgelist(
            PixelDataQuerier(self.view),
            components=self._active_components,
        )

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
            PixelDataQuerier(self.view),
            components=self._active_components,
            markers=self._active_markers,
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
            PixelDataQuerier(self.view),
            components=self._active_components,
            add_marker_counts=add_marker_counts,
            add_spherical_norm=add_spherical_norm,
        )

    def metadata(
        self,
    ) -> dict:
        """Return the metadata for the dataset."""
        return PixelDataQuerier(self.view).read_metadata()

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

        active_samples = samples or self._copy_or_none(self._active_samples)
        active_components = components or self._copy_or_none(self._active_components)
        active_markers = markers or self._copy_or_none(self._active_markers)

        return PNAPixelDataset(
            view=self._view,
            config=copy.copy(self._config),
            active_samples=active_samples,
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
        description += [f"""Mapping the following samples to files:"""]
        for sample_name, file in self.view.sample_to_file_mappings.items():
            description.append(f"\tSample: {sample_name}, File: {file}")

        description.append("")
        adata = self.adata()
        description += [f"""In total it contains:"""]
        description.append(f"{len(adata.obs)} components, {len(adata.var)} markers")
        return "\n".join(description)


class PixelDatasetSaver:
    """A class to save a PixelDataset to disk."""

    def __init__(self, pxl_dataset: PNAPixelDataset):
        """Create a new PixelDatasetSaver instance."""
        self.pxl_dataset = pxl_dataset

    def save(
        self,
        sample_name: str,
        output_path: Path | str,
        optimize_disk_usage: bool = True,
    ) -> PxlFile:
        """Save a sample from a the PixelDataset to disk as a single pxl file with any component filters applied to it.

        NB: for the time being, no marker filters are applied to the saved file.

        This will copy the entire sample to a new file, applying any filters that have been set on the PxlFile
        on-disk.

        :param sample_name: The name of the sample to save.
        :param output_path: The path to save the sample to.
        :param optimize_disk_usage: If True, the saved file will be optimized for disk usage. If this is active
                                    a temporary file will be written before the final file is written to disk.

        :return: The PxlFile pointing to the saved PixelDataset.
        """
        try:
            input_sample = self.pxl_dataset.view.sample_to_file_mappings[sample_name]
        except KeyError:
            raise ValueError(
                f"Sample {sample_name} not found in the PixelDataset. Use one of: {self.pxl_dataset.sample_names()}"
            )

        if isinstance(output_path, str):
            output_path = Path(output_path)

        input_sample_pxl_file = PxlFile(input_sample)

        if optimize_disk_usage:
            with tempfile.NamedTemporaryFile() as temp_file:
                tmp_output_sample = PxlFile.copy_pxl_file(
                    input_sample_pxl_file, Path(temp_file.name)
                )
                InplacePixelDataFilterer(tmp_output_sample).filter_components(
                    self.pxl_dataset.components(),
                    metadata=input_sample_pxl_file.metadata(),
                )
                copy_databases(tmp_output_sample.path, output_path)
                return PxlFile(output_path, sample_name)

        output_sample = PxlFile.copy_pxl_file(input_sample_pxl_file, output_path)
        InplacePixelDataFilterer(output_sample).filter_components(
            self.pxl_dataset.components(),
            metadata=input_sample_pxl_file.metadata(),
        )
        return output_sample
