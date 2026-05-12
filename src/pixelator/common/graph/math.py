"""Graph math utilities.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import numpy as np


def _mat_pow(mat, power: int, prune_threshold: float | None = None):
    """Compute sparse matrix power with optional pruning.

    Args:
        mat: Input sparse matrix.
        power: Exponent to raise the matrix to.
        prune_threshold: Optional absolute-value threshold below which entries
            are set to zero before each multiplication step.

    Returns:
        A sparse matrix equivalent to ``mat`` raised to ``power``.

    """
    mat_power = mat.copy()
    for _ in range(power - 1):
        if prune_threshold:
            mat_power.data[np.abs(mat_power.data) < prune_threshold] = 0
            mat_power.eliminate_zeros()
        mat_power = mat @ mat_power
    return mat_power
