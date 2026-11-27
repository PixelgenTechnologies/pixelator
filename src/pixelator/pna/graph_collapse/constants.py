"""Default parameter values for graph-collapse.

Copyright © 2026 Pixelgen Technologies AB.
"""

# Maximum number of BFS hops from a searched node to look for a collapse candidate.
DEFAULT_BFS_DISTANCE = 6

# Nodes with degree above this value are not searched. 1 investigates only leaves.
DEFAULT_MAX_DEGREE = 1

# Maximum Hamming distance (inclusive) between a searched node's UMI and a candidate's
# UMI for the node to be collapsed into that candidate.
DEFAULT_HAMMING_THRESHOLD = 2

# Components are collapsed and written in batches of about this many edges. A batch
# holds whole components, so a larger component forms a batch of its own.
DEFAULT_BATCH_EDGES = 20_000_000
