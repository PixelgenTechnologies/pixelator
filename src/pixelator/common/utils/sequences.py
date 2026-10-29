"""Utility functions for working with DNA sequences.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

# this tr table is used to complement DNA sequences
_TRTABLE = str.maketrans("GTACN", "CATGN")


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA seq.

    Args:
        seq: the DNA sequence

    Returns:
        the reverse complement of the input sequence (str)
    """
    return seq.translate(_TRTABLE)[::-1]
