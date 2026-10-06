"""Adapt queries so an edgelist with legacy marker column names matches the current schema.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pixelator.pna.pixeldataset.io import PixelDataViewerSession, Query

_LEGACY_MARKER_COLUMNS = (("marker1", "marker_1"), ("marker2", "marker_2"))


class LegacyMarkerQueryAdapter:
    """Rename ``marker1`` and ``marker2`` on queries against a pre-rename edgelist.

    A current file is left unchanged. Construct one adapter per open session and
    reuse it for every query in that session.
    """

    def __init__(self, session: PixelDataViewerSession) -> None:
        """Record which legacy marker columns this session's edgelist still has."""
        columns = set(
            session.execute_eager(
                Query(
                    "SELECT column_name FROM (DESCRIBE SELECT * FROM edgelist)",
                    {},
                )
            )["column_name"].to_list()
        )
        self._rename = ", ".join(
            f"{old} AS {new}" for old, new in _LEGACY_MARKER_COLUMNS if old in columns
        )

    def adapt(self, query: Query) -> Query:
        """Return ``query`` with legacy marker columns renamed, when any are present."""
        if not self._rename:
            return query
        return Query(
            sql=(f"SELECT * RENAME ({self._rename}) FROM ({query.sql}) AS edgelist"),
            params=query.params,
        )
