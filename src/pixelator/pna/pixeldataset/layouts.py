"""On-the-fly layouts wrapper for PNA pixel datasets.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl

from pixelator.common.graph.backends.protocol import (
    DEFAULT_LAYOUT_ALGORITHM,
    SupportedLayoutAlgorithm,
)
from pixelator.pna.pixeldataset.edgelist import Edgelist
from pixelator.pna.pixeldataset.io import PixelDataViewer
from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.utils import normalize_input_to_set


class Layouts:
    """On-the-fly Layouts for one or more Components.

    Coordinates are computed from each Component graph.
    """

    def __init__(
        self,
        view: PixelDataViewer,
        components: str | Iterable[str] | None = None,
        adata_helper: AnnDataHelper | None = None,
        algorithm: SupportedLayoutAlgorithm = DEFAULT_LAYOUT_ALGORITHM,
        add_marker_counts: bool = True,
        add_spherical_norm: bool = False,
        random_seed: int | None = None,
        algorithm_kwargs: dict | None = None,
    ):
        """Create a new Layouts collection."""
        self._view = view
        self._components = normalize_input_to_set(components)
        self._adata_helper = (
            adata_helper
            if adata_helper is not None
            else AnnDataHelper(self._view, components=self._components)
        )
        self._algorithm = algorithm
        self._add_marker_counts = add_marker_counts
        self._add_spherical_norm = add_spherical_norm
        self._random_seed = random_seed
        self._algorithm_kwargs = algorithm_kwargs or {}

    def _obs(self):
        return self._adata_helper.read_adata(
            add_clr_transform=False, add_log1p_transform=False
        ).obs

    def _ordered_components(self) -> list[str]:
        ordered = self._obs().index.to_list()
        if self._components is None:
            return ordered
        return [
            component_id for component_id in ordered if component_id in self._components
        ]

    @property
    def components(self) -> set[str]:
        """Get the component names."""
        return set(self._ordered_components())

    def first(self) -> Layouts:
        """Return a Layouts collection with only the first Component."""
        ordered = self._ordered_components()
        if not ordered:
            raise ValueError("No components available to compute a Layout.")
        return Layouts(
            view=self._view,
            components=[ordered[0]],
            adata_helper=self._adata_helper,
            algorithm=self._algorithm,
            add_marker_counts=self._add_marker_counts,
            add_spherical_norm=self._add_spherical_norm,
            random_seed=self._random_seed,
            algorithm_kwargs=self._algorithm_kwargs,
        )

    def _compute_component(self, component_id: str) -> pl.DataFrame:
        edgelist = Edgelist(
            self._view,
            components=[component_id],
            adata_helper=self._adata_helper,
        )
        component = next(iter(edgelist.iterator()))
        coordinates = component.graph.layout_coordinates(
            layout_algorithm=self._algorithm,
            get_node_marker_matrix=self._add_marker_counts,
            random_seed=self._random_seed,
            **self._algorithm_kwargs,
        )
        sample = self._obs().loc[component_id, "sample"]
        coordinates = coordinates.reset_index(drop=True)
        coordinates["component"] = component_id
        coordinates["graph_projection"] = "full"
        coordinates["layout"] = self._algorithm
        coordinates["sample"] = sample
        return self._post_process(pl.from_pandas(coordinates))

    def _post_process(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._add_spherical_norm:
            coord_cols = [c for c in ("x", "y", "z") if c in df.columns]
            coordinates = df.select(coord_cols).to_numpy()
            normalized_coordinates = pl.DataFrame(
                coordinates / (1 * np.linalg.norm(coordinates, axis=1))[:, None],
                schema={f"{c}_norm": pl.Float32 for c in coord_cols},
            )
            df = df.hstack(normalized_coordinates)
        return df

    def to_polars(self) -> pl.DataFrame:
        """Get the Layouts as a polars DataFrame."""
        frames = [frame for _, frame in self.iterator(return_polars_df=True)]
        if not frames:
            return pl.DataFrame()
        df = pl.concat(frames, how="diagonal_relaxed")
        if self._add_marker_counts:
            df = df.with_columns(pl.selectors.unsigned_integer().fill_null(0))
        return df

    def to_df(self) -> pd.DataFrame:
        """Get the Layouts as a pandas DataFrame."""
        return self.to_polars().to_pandas()

    def iterator(
        self, return_polars_df: bool = False
    ) -> Iterable[tuple[str, pd.DataFrame | pl.DataFrame]]:
        """Yield (component id, Layout dataframe) pairs, one Component at a time."""
        for component_id in self._ordered_components():
            frame = self._compute_component(component_id)
            if return_polars_df:
                yield component_id, frame
            else:
                yield component_id, frame.to_pandas()

    def is_empty(self) -> bool:
        """Return True when there are no Components to layout."""
        return not self._ordered_components()

    def describe(self) -> str:
        """Return a description of the Layouts."""
        return (
            f"Layouts({len(self._ordered_components()):,} components, "
            f"algorithm={self._algorithm})"
        )

    def __str__(self):
        """Get a string representation of the Layouts."""
        return (
            f"Layouts({len(self._ordered_components()):,} components, "
            f"algorithm={self._algorithm})"
        )

    def __repr__(self):
        """Get a string representation of the Layouts."""
        return str(self)

    def _ipython_display_(self):
        return print(self.describe())
