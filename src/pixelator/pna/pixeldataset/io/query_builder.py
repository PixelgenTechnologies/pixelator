"""Query building utilities for PXL-backed pixeldatasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sized

from pixelator.pna.analysis.proximity import jcs_with_analytical_stats


@dataclass(frozen=True, slots=True)
class Query:
    """Encapsulate SQL text and bound parameters."""

    sql: str
    params: dict[str, object]


class QueryBuilder:
    """Build `Query` objects for pixeldataset read operations."""

    @staticmethod
    def _optimized_where_condition(
        column_name: str, param_name: str, values: Sized | None
    ) -> str:
        if not values:
            return "TRUE"
        if len(values) == 1:
            return f"{column_name} = ${param_name}"
        return f"{column_name} IN ${param_name}"

    @staticmethod
    def _normalize_list_param(
        param_name: str, values: list[str] | None
    ) -> dict[str, object]:
        if values is None:
            return {}
        return {param_name: values if len(values) > 1 else values[0]}

    @staticmethod
    def _optimized_component_where_condition(components: Sized | None) -> str:
        if not components:
            return "TRUE"
        if len(components) == 1:
            return "component = $components"
        return "component IN $components"

    @staticmethod
    def _normalize_components_param(
        components: list[str] | None,
    ) -> dict[str, object]:
        if components is None:
            return {}
        return {
            "components": components if len(components) > 1 else components[0],
        }

    def adata_X_query(
        self,
        db_name: str,
        components: list[str] | None = None,
    ) -> Query:
        """Build AnnData X query for a single underlying sample DB.

        Note: marker filtering is intentionally not pushed down into SQL (column
        selection); consumers can filter markers in-memory after materialization.
        """
        return Query(
            sql=f"""SELECT * FROM {db_name}.__adata__X
                    WHERE {self._optimized_where_condition("index", "components", components)}
                """,
            params=self._normalize_list_param("components", components),
        )

    def adata_obs_query(
        self,
        db_name: str,
        components: list[str] | None = None,
    ) -> Query:
        """Build AnnData obs query for a single underlying sample DB."""
        return Query(
            sql=f"""SELECT * FROM {db_name}.__adata__obs
                    WHERE {self._optimized_where_condition("index", "components", components)}
                """,
            params=self._normalize_list_param("components", components),
        )

    def adata_var_query(
        self,
        db_name: str,
        markers: list[str] | None = None,
    ) -> Query:
        """Build AnnData var query for a single underlying sample DB."""
        return Query(
            sql=f"""SELECT * FROM {db_name}.__adata__var
                    WHERE {self._optimized_where_condition("index", "markers", markers)}
                """,
            params=self._normalize_list_param("markers", markers),
        )

    def adata_uns_query(self, db_name: str) -> Query:
        """Build AnnData uns query for a single underlying sample DB."""
        return Query(
            sql=f"SELECT * FROM {db_name}.__adata__uns",
            params={},
        )

    def adata_obsm_table_names_query(self, db_name: str) -> Query:
        """Build query listing available obsm tables for a sample DB."""
        return Query(
            sql="SHOW ALL TABLES",
            params={},
        )

    def adata_obsm_query(
        self,
        db_name: str,
        table_name: str,
        components: list[str] | None = None,
    ) -> Query:
        """Build AnnData obsm query for a given obsm table in a sample DB."""
        qualified = table_name if "." in table_name else f"{db_name}.main.{table_name}"
        return Query(
            sql=f"""SELECT * FROM {qualified}
                    WHERE {self._optimized_where_condition("index", "components", components)}
                """,
            params=self._normalize_list_param("components", components),
        )

    def edgelist_query(self, components: list[str] | None) -> Query:
        """Build an edgelist data query."""
        return Query(
            sql=f"""SELECT * FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
                """,
            params=self._normalize_components_param(components),
        )

    def edgelist_len_query(self, components: list[str] | None) -> Query:
        """Build an edgelist count query."""
        return Query(
            sql=f"""SELECT COUNT(*) FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
            """,
            params=self._normalize_components_param(components),
        )

    def layouts_query(
        self, components: list[str] | None, add_marker_counts: bool
    ) -> Query:
        """Build a layouts query, optionally including marker-count join data."""
        if add_marker_counts:
            return Query(
                sql=f"""
                WITH filtered_edgelist AS (
                    SELECT umi1 as umi, marker_1 as marker
                    FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}

                    UNION

                    SELECT umi2 as umi, marker_2 as marker
                    FROM edgelist
                    WHERE {self._optimized_component_where_condition(components)}
                )
                SELECT *
                FROM layouts
                LEFT JOIN filtered_edgelist as umi_and_markers
                ON layouts.index = umi_and_markers.umi
                WHERE {self._optimized_component_where_condition(components)}
            """,
                params=self._normalize_components_param(components),
            )
        return Query(
            sql=f"""SELECT * FROM layouts
                        WHERE {self._optimized_component_where_condition(components)}
                    """,
            params=self._normalize_components_param(components),
        )

    def layouts_len_query(self, components: list[str] | None) -> Query:
        """Build a layouts count query."""
        return Query(
            sql=f"""SELECT COUNT(*)
                    FROM layouts
                    WHERE {self._optimized_component_where_condition(components)}
                """,
            params=self._normalize_components_param(components),
        )

    def proximity_query(
        self,
        components: list[str] | None,
        markers: list[str] | None,
        calculate_from_edgelist: bool = False,
    ) -> Query:
        """Build a proximity data query."""
        if calculate_from_edgelist:
            sql, params = jcs_with_analytical_stats(
                components=components,
                markers=markers,
            )
        else:
            sql = f"""SELECT * FROM proximity
                        WHERE  {self._optimized_component_where_condition(components)} AND
                                {"(marker_1 IN $markers AND marker_2 IN $markers)" if markers else "TRUE"};
                    """
            params = self._normalize_components_param(components)
            if markers is not None:
                params["markers"] = markers
        return Query(sql=sql, params=params)

    def proximity_len_query(
        self,
        components: list[str] | None,
        markers: list[str] | None,
        calculate_from_edgelist: bool = False,
    ) -> Query:
        """Build a proximity count query."""
        if calculate_from_edgelist:
            return self._calculate_proximity_length_from_edgelist(components, markers)

        else:
            params = self._normalize_components_param(components)
            if markers is not None:
                params["markers"] = markers
            return Query(
                sql=f"""SELECT COUNT(*) FROM proximity
                            WHERE {self._optimized_component_where_condition(components)} AND
                                {"(marker_1 IN $markers AND marker_2 IN $markers)" if markers else "TRUE"};
                        """,
                params=params,
            )

    def _calculate_proximity_length_from_edgelist(
        self, components: list[str] | None, markers: list[str] | None
    ) -> Query:
        params = self._normalize_components_param(components)
        if markers is not None:
            params["markers"] = markers

        query = f"""
        SELECT CAST(SUM(marker_count * (marker_count+1) / 2) AS INTEGER) AS count
        FROM (
            SELECT component, COUNT(DISTINCT marker) AS marker_count
            FROM (
                SELECT component, marker_1 AS marker FROM edgelist
                UNION
                SELECT component, marker_2 AS marker FROM edgelist
            )
            WHERE {self._optimized_component_where_condition(components)} AND
                    {"(marker IN $markers)" if markers else "TRUE"}
            GROUP BY component
        )
        """
        return Query(sql=query, params=params)
