"""Top level module for analysis subpackage of pixelator.pna.

Copyright © 2024 Pixelgen Technologies AB.
"""

from pixelator.pna.analysis._differential_abundance import differential_abundance
from pixelator.pna.analysis.proximity import (
    calculate_differential_proximity,
    filter_proximity_scores,
    summarize_proximity_scores,
)
from pixelator.pna.analysis.segmentation import (
    cc_protein_weights,
    distance_from_node_set,
    partition_counts,
    segment_cell,
)

__all__ = [
    "calculate_differential_proximity",
    "cc_protein_weights",
    "differential_abundance",
    "distance_from_node_set",
    "filter_proximity_scores",
    "partition_counts",
    "segment_cell",
    "summarize_proximity_scores",
]

# Note: pixelator.pna.analysis.comparison is intentionally not imported here.
# It depends on pixelator.pna.pixeldataset, which itself imports
# pixelator.pna.analysis.analytical_proximity_query_helper during its own
# initialization. Eagerly importing comparison here would create a circular
# import. Import it directly, e.g. `from pixelator.pna.analysis.comparison
# import compare_sample_pairs`.
