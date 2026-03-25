"""Shared small types for PNA pixel datasets.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pixelator.pna.graph import PNAGraph, PNAGraphBackend


@dataclass(slots=True, frozen=True, repr=True)
class Component:
    """A dataclass to hold a component and its associated graph."""

    component_id: str
    frame: pl.LazyFrame

    @property
    def graph(self) -> PNAGraph:
        """Get the graph."""
        return PNAGraph(PNAGraphBackend.from_edgelist(self.frame))
