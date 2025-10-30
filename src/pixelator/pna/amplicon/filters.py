"""Filter classes for the PNA assay pipeline.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import Optional

import numpy as np
from cutadapt._align import Aligner
from cutadapt.adapters import Where
from cutadapt.info import ModificationInfo
from cutadapt.predicates import Predicate
from cutadapt.steps import SingleEndFilter
from dnaio._core import SequenceRecord

from pixelator.pna.config import PNAAssay, get_position_in_parent


class SingleEndFilterWithFailureReason(SingleEndFilter):
    """A single-end read filter that records failure reasons.

    A pipeline step that can filter reads, can redirect filtered ones to a writer, and
    counts how many were filtered.

    The "descriptive_identifier" method is used to provide a string that will be appended to
    the read identifier of filtered reads when a filtered writer is provided.
    """

    def __init__(
        self,
        predicate: Predicate,
        writer=None,
    ):
        """Initialize a SingleEndFilterWithFailureReason pipeline step."""
        self._filtered = 0
        self._predicate = predicate
        self._writer = writer

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"SingleEndFilter(predicate={self._predicate}, writer={self._writer})"

    def descriptive_identifier(self) -> str:
        """Return a string identifier for this predicate.

        Used in reports and added as a comment to the sequence id when writing failed reads.
        """
        return self._predicate.descriptive_identifier()

    def filtered(self) -> int:
        """Return the number of filtered reads."""
        return self._filtered

    def __call__(self, read, info: ModificationInfo) -> Optional[SequenceRecord]:
        """Filter a single-end read.

        The implementation is identical to the normal SingleEndFilter except that the read id
        is modified by appending the `descriptive_identifier` of the filter step.

        Args:
            read: The read to filter.
            info: The modification info.

        """
        if self._predicate.test(read, info):
            self._filtered += 1
            if self._writer is not None:
                read.name += f" {self.descriptive_identifier()}"
                self._writer.write(read)
            return None
        return read


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

    def descriptive_identifier(self) -> str:
        """Return a string identifier for this predicate.

        Used in reports and added as a comment to the sequence id when writing failed reads.
        """
        return "too_many_n"

    def __repr__(self):
        """Return a string representation of the object."""
        return f"TooManyN(cutoff={self.cutoff}, is_proportion={self.is_proportion})"

    def test(self, read, info: ModificationInfo):
        """Return True if the read has too many N bases.

        Args:
            read: The read to test.
            info: The modification info.

        Returns:
            True if the read has too many N bases, False otherwise.

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


class LowComplexityUMI(Predicate):
    """Select reads that have low complexity in the UMI1 and UMI2 regions.

    Low complexity is defined as having a single base that makes up more than
    a given proportion of the bases in the read.

    This filtering expects the full amplicon sequence as input.
    """

    def __init__(self, assay: PNAAssay, proportion: float = 0.80) -> None:
        """Initialize a LowComplexityFilter pipeline predicate.

        Args:
            assay: the assay configuration.
            proportion: the proportion of a single base that defines low complexity.

        """
        if not 0.0 < proportion < 1.0:
            raise ValueError(
                f"proportion must be between 0.0 and 1.0, got {proportion}."
            )

        self.proportion = proportion
        self.assay = assay

        uei_pos = get_position_in_parent(self.assay, "uei")
        self._umi1_region_slice = slice(*get_position_in_parent(assay, "umi-1"))
        self._umi2_region_slice = slice(*get_position_in_parent(assay, "umi-2"))

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"LowComplexityFilter(proportion={self.proportion})"

    def descriptive_identifier(cls) -> str:
        """Return a string identifier for this predicate.

        Used in reports and added as a comment to the sequence id when writing failed reads.
        """
        return "low_complexity_umi"

    def test(self, read, info: ModificationInfo = None) -> bool:
        """Return True if the read has low complexity.

        Args:
            read: The read to test.
            info: The modification info.

        Returns:
            True if the read has low complexity, False otherwise.

        """
        umi1 = np.frombuffer(
            read.sequence[self._umi1_region_slice].encode("ascii"), dtype="S1"
        )
        umi2 = np.frombuffer(
            read.sequence[self._umi2_region_slice].encode("ascii"), dtype="S1"
        )

        _, umi1_counts = np.unique(umi1, return_counts=True)
        _, umi2_counts = np.unique(umi2, return_counts=True)

        umi1_char_freqs = umi1_counts / len(umi1)
        umi2_char_freqs = umi2_counts / len(umi2)

        if np.any(umi1_char_freqs > self.proportion):
            return True

        if np.any(umi2_char_freqs > self.proportion):
            return True

        return False


class LBSDetectedInUMI(Predicate):
    """Select wrongly anchored-reads that match a part of the LBS regions in the UMIs.

    In UMI1 we check for matches to LBS1 at the end of the UMI1 region.
    In UMI2 we check for matches to LBS2 at the start of the UMI2 region.

    This filtering step expects the full amplicon sequence as input.

    Args:
        assay: The assay configuration.
        min_overlap: Minimum overlap for alignment.
        max_error_rate: Maximum error rate for alignment.

    """

    def __init__(self, assay, min_overlap=8, max_error_rate=0.125) -> None:
        """Initialize a LBSDetectedInUMI pipeline predicate."""
        self.assay = assay
        self.min_overlap = min_overlap
        self.max_error_rate = max_error_rate

        lbs1_ref = assay.get_region_by_id("lbs-1").get_sequence()
        lbs2_ref = assay.get_region_by_id("lbs-2").get_sequence()

        self._lbs1_aligner = Aligner(
            reference=lbs1_ref,
            max_error_rate=self.max_error_rate,
            wildcard_ref=False,
            wildcard_query=True,
            min_overlap=min_overlap,
            flags=Where.ANYWHERE,
        )

        self._lbs2_aligner = Aligner(
            reference=lbs2_ref,
            max_error_rate=self.max_error_rate,
            wildcard_ref=False,
            wildcard_query=True,
            min_overlap=min_overlap,
            flags=Where.ANYWHERE,
        )

        self._umi1_region_slice = slice(*get_position_in_parent(assay, "umi-1"))
        self._umi2_region_slice = slice(*get_position_in_parent(assay, "umi-2"))

    def __repr__(self):
        """Return a string representation of the object."""
        return f"LBSDetectedInUMI(min_overlap={self.min_overlap}, max_error_rate={self.max_error_rate})"

    def descriptive_identifier(cls) -> str:
        """Return a string identifier for this predicate.

        Used in reports and added as a comment to the sequence id when writing failed reads.
        """
        return "lbs_detected_in_umi"

    def test(self, read, info: ModificationInfo = None) -> bool:
        """Return True if the read contains any fixed region substrings.

        Args:
            read: The read to test.
            info: The modification info.

        Returns:
            True if the read contains any fixed region substrings, False otherwise.

        """
        umi1 = read.sequence[self._umi1_region_slice]
        umi2 = read.sequence[self._umi2_region_slice]

        alm1 = self._lbs1_aligner.locate(umi1)
        alm2 = self._lbs2_aligner.locate(umi1)

        if alm1 is not None or alm2 is not None:
            return True

        alm1 = self._lbs1_aligner.locate(umi2)
        alm2 = self._lbs2_aligner.locate(umi2)

        if alm1 is not None or alm2 is not None:
            return True

        return False
