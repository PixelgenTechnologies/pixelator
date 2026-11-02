"""Graph-collapse: a post-graph-step alternative to correcting UMI sequencing errors.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pixelator.pna.graph_collapse.collapse import (
    GraphCollapseResult,
    run_graph_collapse,
)

__all__ = [
    "GraphCollapseResult",
    "run_graph_collapse",
]
