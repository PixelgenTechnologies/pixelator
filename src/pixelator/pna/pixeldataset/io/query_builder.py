"""Query building utilities for PXL-backed pixeldatasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sized


@dataclass(frozen=True, slots=True)
class Query:
    """Encapsulate SQL text and bound parameters."""

    sql: str
    params: dict[str, object]


class QueryBuilder:
    """Build `Query` objects for pixeldataset read operations."""

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
        self, components: list[str] | None, markers: list[str] | None
    ) -> Query:
        """Build a proximity data query."""
        params = self._normalize_components_param(components)
        if markers is not None:
            params["markers"] = markers
        return Query(
            sql=f"""SELECT * FROM proximity
                         WHERE  {self._optimized_component_where_condition(components)} AND
                                {"(marker_1 IN $markers AND marker_2 IN $markers)" if markers else "TRUE"};
                    """,
            params=params,
        )

    def proximity_len_query(
        self, components: list[str] | None, markers: list[str] | None
    ) -> Query:
        """Build a proximity count query."""
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
