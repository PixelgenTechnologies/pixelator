"""Filter classes for the PNA assay pipeline.

Copyright © 2024 Pixelgen Technologies AB.
"""

from cutadapt.info import ModificationInfo
from cutadapt.predicates import Predicate

from pixelator.pna.config import PNAAssay, get_position_in_parent


class TooManyN(Predicate):
    """Select reads that have too many 'N' bases.

    Both a raw count or a proportion (relative to the sequence length) can be used.
    """

    def __init__(self, count: float, assay: PNAAssay):
        """Initialize a TooManyN pipeline predicate.

        :param count: the cutoff for the N count.
            If it is below 1.0, it will be considered a proportion, and above and equal to
            1 will be considered as discarding reads with a number of N's greater than this cutoff.
        :param assay: the assay configuration.
        """
        assert count >= 0
        self.is_proportion = count < 1.0
        self.cutoff = count
        self.assay = assay
        uei_pos = get_position_in_parent(self.assay, "uei")
        self._region1 = slice(0, uei_pos[0])
        self._region2 = slice(uei_pos[1], -1)

    def __repr__(self):
        """Return a string representation of the object."""
        return f"TooManyN(cutoff={self.cutoff}, is_proportion={self.is_proportion})"

    def test(self, read, info: ModificationInfo):
        """Return True if the read has too many N bases.

        :param read: the read to test
        :param info: the modification info
        """
        r = read.sequence.lower()
        region1 = r[self._region1]
        region2 = r[self._region2]

        read_len = len(region1) + len(region2)
        n_count = region1.count("n") + region2.count("n")
        if self.is_proportion:
            if len(read) == 0:
                return False
            return n_count / read_len > self.cutoff
        else:
            return n_count > self.cutoff
