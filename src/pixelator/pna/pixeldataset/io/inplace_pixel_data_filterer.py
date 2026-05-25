"""In-place component filtering for PXL files.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import duckdb

from pixelator.pna.pixeldataset.io.anndata_helper import AnnDataHelper
from pixelator.pna.pixeldataset.io.pixel_data_viewer import PixelDataViewer
from pixelator.pna.pixeldataset.io.pixel_file_writer import PixelFileWriter
from pixelator.pna.pixeldataset.io.pxl_file import PxlFile
from pixelator.pna.utils.utils import normalize_input_to_list


class InplacePixelDataFilterer:
    """Class to filter a PXL file in place.

    This is mostly useful for testing purposes, when one wants to strip
    components from a PXL file to make it smaller and faster to work with.
    """

    def __init__(self, pxl_file: PxlFile):
        """Initialize the InplacePixelDataFilterer.

        Args:
            pxl_file: Pxl file.
        """
        self.pxl_file = pxl_file

    def _update_metadata(
        self, connection: duckdb.DuckDBPyConnection, metadata: dict
    ) -> None:
        connection.sql(
            """
            DROP TABLE IF EXISTS metadata;
            CREATE TABLE metadata (value JSON);
            INSERT INTO metadata VALUES ($metadata);
            """,
            params={"metadata": metadata},
        )

    def _filter_edgelist(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        query = f"""
            DELETE FROM edgelist
            WHERE component NOT IN $components
        """
        connection.sql(query, params={"components": components})

    def _filter_proximity(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        try:
            query = f"""
                DELETE FROM proximity
                WHERE component NOT IN $components
            """
            connection.sql(query, params={"components": components})
        except duckdb.CatalogException:
            pass

    def _filter_layouts(
        self, connection: duckdb.DuckDBPyConnection, components: list[str]
    ) -> None:
        try:
            query = f"""
                DELETE FROM layouts
                WHERE component NOT IN $components
            """
            connection.sql(query, params={"components": components})
        except duckdb.CatalogException:
            pass

    def _filter_adata(self, pxl_file: PxlFile, components: list[str]) -> None:
        adata = AnnDataHelper(PixelDataViewer.from_files([pxl_file])).read_adata(
            add_clr_transform=False, add_log1p_transform=False
        )
        adata = adata[adata.obs.index.isin(components)]
        with PixelFileWriter(pxl_file.path) as writer:
            writer.write_adata(adata)

    def filter_components(
        self, components: set[str] | list[str], metadata: dict | None = None
    ) -> None:
        """Filter the PXL file by components, only keeping the provided components.

        Note that if you provide metadata it will overwrite the existing metadata.
        If you do not provide metadata, the existing metadata will be kept.

        Args:
            components: The components to keep.
            metadata: The metadata to write to the PXL
        """
        # If there are not components provided, do nothing.
        if not components:
            raise ValueError("You must provided at least one component to filter.")

        components_as_list: list[str] = normalize_input_to_list(components)  # type: ignore

        with duckdb.connect(self.pxl_file.path) as connection:
            self._filter_edgelist(connection, components_as_list)
            self._filter_proximity(connection, components_as_list)
            self._filter_layouts(connection, components_as_list)
            if metadata:
                self._update_metadata(connection, metadata)

        self._filter_adata(self.pxl_file, components_as_list)
