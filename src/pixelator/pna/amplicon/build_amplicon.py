"""Build an amplicon from paired-end reads.

Copyright © 2024 Pixelgen Technologies AB.
"""

import enum
import typing
from typing import Optional, cast

import numpy as np
from cutadapt._align import Aligner
from cutadapt.adapters import Where
from cutadapt.info import ModificationInfo
from cutadapt.statistics import ReadLengthStatistics
from cutadapt.steps import HasFilterStatistics
from dnaio import SequenceRecord

from pixelator.pna.config import SequenceType
from pixelator.pna.config.assay import PNAAssay, Region, get_position_in_parent
from pixelator.pna.read_processing.modifiers import CombiningModifier
from pixelator.pna.read_processing.statistics import HasCustomStatistics

#: Translation table for reverse complementing DNA sequences (strings)
_TRTABLE = str.maketrans("GTACN", "CATGN")

#: Translation table for reverse complementing DNA sequences (bytes)
_TRTABLE_BYTES = bytes.maketrans(b"GTACN", b"CATGN")

BytesOrString = typing.TypeVar("BytesOrString", str, bytes)


def reverse_complement(seq: BytesOrString) -> BytesOrString:
    """Compute the reverse complement of a DNA seq.

    :param seq: the DNA sequence
    :return: the reverse complement of the input sequence
    :rtype: Bytes or Str depending on the input
    """
    if isinstance(seq, bytes):
        return seq.translate(_TRTABLE_BYTES)[::-1]

    return seq.translate(_TRTABLE)[::-1]


class AmpliconBuilderStatistics:
    """Helper class for AmpliconBuilder to keep track of pipeline statistics."""

    def __init__(self):
        """Initialize the statistics."""
        self.passed_reads = 0
        self.failed_reads = 0

        self.passed_missing_lbs1_anchor = 0
        self.passed_partial_uei_reads = 0
        self.failed_partial_upi1_umi1_reads = 0
        self.failed_partial_upi2_umi2_reads = 0

        self.passed_missing_uei_reads = 0
        self.failed_missing_upi1_umi1_reads = 0
        self.failed_missing_upi2_umi2_reads = 0

    def __iadd__(self, other):
        """Merge statistics from another object into this one."""
        if not isinstance(other, self.__class__):
            raise ValueError("Cannot compare")

        self.passed_reads += other.passed_reads
        self.failed_reads += other.failed_reads

        self.passed_missing_lbs1_anchor += other.passed_missing_lbs1_anchor
        self.passed_partial_uei_reads += other.passed_partial_uei_reads
        self.failed_partial_upi1_umi1_reads += other.failed_partial_upi1_umi1_reads
        self.failed_partial_upi2_umi2_reads += other.failed_partial_upi2_umi2_reads

        self.passed_missing_uei_reads += other.passed_missing_uei_reads
        self.failed_missing_upi1_umi1_reads += other.failed_missing_upi1_umi1_reads
        self.failed_missing_upi2_umi2_reads += other.failed_missing_upi2_umi2_reads

        return self

    def collect(self) -> dict[str, int]:
        """Return a dictionary with statistics."""
        return {
            "passed_reads": self.passed_reads,
            "passed_missing_eui_reads": self.passed_missing_uei_reads,
            "failed_reads": self.failed_reads,
            "passed_partial_uei_reads": self.passed_partial_uei_reads,
            "passed_missing_lbs1_anchor": self.passed_missing_lbs1_anchor,
            "failed_partial_upi1_umi1_reads": self.failed_partial_upi1_umi1_reads,
            "failed_partial_upi2_umi2_reads": self.failed_partial_upi2_umi2_reads,
            "failed_missing_upi1_umi1_reads": self.failed_missing_upi1_umi1_reads,
            "failed_missing_upi2_umi2_reads": self.failed_missing_upi2_umi2_reads,
        }

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"AmpliconBuilderStatistics(passed_reads={self.passed_reads}, failed_reads={self.failed_reads})"


class AmpliconCombiner:
    """Helper class for `AmpliconBuilder` to assemble an amplicon from different regions."""

    _REQUIRED_REGIONS = {
        "amplicon",
        "lbs-1",
        "lbs-2",
        "pid-1",
        "pid-2",
        "umi-1",
        "umi-2",
        "uei",
    }

    def __init__(self, assay: PNAAssay):
        """Initialize the AmpliconCombiner.

        :param assay: the assay design
        """
        self.assay = assay
        self._template = (
            cast(Region, assay.get_region_by_id("amplicon"))
            .get_sequence()
            .encode("ascii")
        )
        self._qualities_template = bytes(b"!" * len(self._template))
        self._uei_template = bytes(b"N" * assay.get_region_by_id("uei").min_len)
        self._uei_qualities_template = bytes(b"!" * len(self._uei_template))

        lbs1_len = cast(Region, assay.get_region_by_id("lbs-1")).min_len
        pid1_len = cast(Region, assay.get_region_by_id("pid-1")).min_len
        umi1_len = cast(Region, assay.get_region_by_id("umi-1")).min_len
        uei_len = cast(Region, assay.get_region_by_id("uei")).min_len
        lbs2_len = cast(Region, assay.get_region_by_id("lbs-2")).min_len
        pid2_len = cast(Region, assay.get_region_by_id("pid-2")).min_len
        umi2_len = cast(Region, assay.get_region_by_id("umi-2")).min_len

        self._pid1_umi1_region_slice = slice(0, pid1_len + umi1_len)
        self._uei_region_slice = slice(
            pid1_len + umi1_len + lbs1_len, pid1_len + umi1_len + lbs1_len + uei_len
        )
        self._pid2_umi2_region_slice = slice(
            pid1_len + umi1_len + lbs1_len + uei_len + lbs2_len,
            pid1_len + umi1_len + lbs1_len + uei_len + lbs2_len + pid2_len + umi2_len,
        )
        self._lbs1_region_slice = slice(*get_position_in_parent(assay, "lbs-1"))
        self._lbs2_region_slice = slice(*get_position_in_parent(assay, "lbs-2"))

    def build_sequence(
        self, pid1_umi1_region: bytes, pid2_umi2_region: bytes, uei_region: bytes | None
    ) -> bytes:
        """Build the amplicon sequence from the different regions.

        :param pid1_umi1_region: the PBS-1 and UMI-1 region
        :param uei_region: the UEI region
        :param pid2_umi2_region: the PBS-2 and UMI-2 region
        """
        # We are using bytearray here since strings are immutable
        s = bytearray(self._template)
        s[self._pid1_umi1_region_slice] = pid1_umi1_region
        s[self._pid2_umi2_region_slice] = pid2_umi2_region

        if uei_region:
            uei_len = len(self._uei_template)
            insert_len = len(uei_region)
            if insert_len >= uei_len:
                s[self._uei_region_slice] = uei_region[:uei_len]
            else:
                s[self._uei_region_slice] = self._uei_template
                s[
                    self._uei_region_slice.start : self._uei_region_slice.start
                    + insert_len
                ] = uei_region
        else:
            s[self._uei_region_slice] = self._uei_template

        # LBS-1 and LBS-2 are already included in the template
        return s

    def build_qualities(
        self,
        pid1_umi1_region: bytes,
        pid2_umi2_region: bytes,
        uei_region: bytes | None = None,
        lbs1_region: bytes | None = None,
        lbs2_region: bytes | None = None,
    ) -> bytes:
        """Build the amplicon qualities from the different regions.

        All regions are assumed to have the correct length.

        :param pid1_umi1_region: the PBS-1 and UMI-1 region qualities
        :param uei_region: the UEI region qualities
        :param pid2_umi2_region: the PBS-2 and UMI-2 region qualities
        :param lbs1_region: the LBS-1 region qualities
        :param lbs2_region: the LBS-2 region qualities
        """
        # We are using bytearray here since strings are immutable
        s = bytearray(self._qualities_template)
        s[self._pid1_umi1_region_slice] = pid1_umi1_region
        s[self._pid2_umi2_region_slice] = pid2_umi2_region

        if uei_region:
            uei_len = len(self._uei_template)
            insert_len = len(uei_region)
            if insert_len >= uei_len:
                s[self._uei_region_slice] = uei_region[:uei_len]
            else:
                s[self._uei_region_slice] = self._uei_qualities_template
                s[
                    self._uei_region_slice.start : self._uei_region_slice.start
                    + insert_len
                ] = uei_region
        else:
            s[self._uei_region_slice] = self._uei_qualities_template

        if lbs1_region is not None:
            assert len(lbs1_region) == (
                self._lbs1_region_slice.stop - self._lbs1_region_slice.start
            )
            s[self._lbs1_region_slice] = lbs1_region
        if lbs2_region is not None:
            assert len(lbs2_region) == (
                self._lbs2_region_slice.stop - self._lbs2_region_slice.start
            )
            s[self._lbs2_region_slice] = lbs2_region

        return s


class AmpliconRegionSlices:
    """Helper class to store slices for the different regions of a PNA amplicon."""

    __slots__ = ("pid1_umi1", "pid2_umi2", "uei", "lbs1", "lbs2")

    def __init__(self):
        """Initialize the AmpliconRegionSlices instance."""
        self.pid1_umi1 = None
        self.pid2_umi2 = None
        self.uei = None
        self.lbs1 = None
        self.lbs2 = None

    def __getitem__(self, item):
        """Provide tuple like access to the slices."""
        if isinstance(item, int):
            key = self.__slots__[item]
            return getattr(self, key)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f"AmpliconRegionSlices(pid1_umi1={self.pid1_umi1}, pid2_umi2={self.pid2_umi2}, "
            f"uei={self.uei}, lbs1={self.lbs1}, lbs2={self.lbs2})"
        )


class AmpliconBuilderFailureReason(enum.Enum):
    """Reasons for failing to build an amplicon."""

    MISSING_PID1_UMI1 = "missing_pid1_umi1"
    MISSING_PID2_UMI2 = "missing_pid2_umi2"
    MISSING_UEI = "missing_uei"
    PARTIAL_PID1_UMI1 = "partial_pid1_umi1"
    PARTIAL_PID2_UMI2 = "partial_pid2_umi2"
    PARTIAL_UEI = "partial_uei"


class AmpliconBuilder(CombiningModifier, HasFilterStatistics, HasCustomStatistics):
    """Construct an amplicon from a pair of reads.

    :param assay: the assay design
    :param mismatches: the maximum number of mismatches allowed when aligning the LBS sequences
    :param writer: a writer to save failed reads to
    """

    def __init__(self, assay: PNAAssay, mismatches: int | float = 0.2, writer=None):
        """Initialize the AmpliconBuilder.

        :param assay: the assay design
        :param mismatches: the maximum number of mismatches allowed when aligning the LBS sequences.
            If a float, it is interpreted as a fraction of the template length. Otherwise, it is
            the maximum number of mismatches allowed.
        :param writer: a writer to save failed reads to.
        """
        self._assay = assay
        self._writer = writer
        self._region_combiner = AmpliconCombiner(assay)

        amplicon_len = assay.get_region_by_id("amplicon").min_len
        lbs1_ref = assay.get_region_by_id("lbs-1").get_sequence()
        lbs2_ref = assay.get_region_by_id("lbs-2").get_sequence()
        lbs1_ref_rc = reverse_complement(lbs1_ref)
        lbs2_ref_rc = reverse_complement(lbs2_ref)

        min_overlap = 4
        indel_cost = 2

        self._lbs1_aligner = Aligner(
            reference=lbs1_ref,
            max_error_rate=mismatches,
            wildcard_ref=False,
            wildcard_query=True,
            indel_cost=indel_cost,
            min_overlap=min_overlap,
            flags=Where.BACK,
        )

        self._lbs2_aligner = Aligner(
            reference=lbs2_ref,
            max_error_rate=mismatches,
            wildcard_ref=False,
            wildcard_query=True,
            indel_cost=indel_cost,
            min_overlap=min_overlap,
            flags=Where.BACK,
        )

        self._lbs1_rc_aligner = Aligner(
            reference=lbs1_ref_rc,
            max_error_rate=mismatches,
            wildcard_ref=False,
            wildcard_query=True,
            indel_cost=indel_cost,
            min_overlap=min_overlap,
            flags=Where.BACK,
        )

        self._lbs2_rc_aligner = Aligner(
            reference=lbs2_ref_rc,
            max_error_rate=mismatches,
            wildcard_ref=False,
            wildcard_query=True,
            indel_cost=indel_cost,
            min_overlap=min_overlap,
            flags=Where.FRONT,
        )

        # Some helper variables
        # We define these here to save a bit of time by not regenerating them for each read
        self._pid_1_umi_1_region_len = (
            assay.get_region_by_id("pid-1").min_len
            + assay.get_region_by_id("umi-1").min_len
        )
        self._pid_2_umi_2_region_len = (
            assay.get_region_by_id("pid-2").min_len
            + assay.get_region_by_id("umi-2").min_len
        )
        self._lbs1_seq = assay.get_region_by_id("lbs-1").get_sequence().encode("ascii")
        self._lbs2_seq = assay.get_region_by_id("lbs-2").get_sequence().encode("ascii")
        self._lbs1_qual = bytes(b"!" * len(self._lbs1_seq))
        self._lbs2_qual = bytes(b"!" * len(self._lbs2_seq))

        self._lbs2_start_pos = get_position_in_parent(assay, "lbs-2")[0]

        self._uei_region_len = assay.get_region_by_id("uei").min_len
        self._uei_region_end_pos = get_position_in_parent(assay, "uei")[1]
        self._uei_region_end_pos_rev = amplicon_len - self._uei_region_end_pos

        # If mismatches is a float, interpret it as a fraction of the template length
        # otherwise it is the maximum edit distance

        self._stats = ReadLengthStatistics()
        self._custom_stats = AmpliconBuilderStatistics()

        self._fixed_regions = self._get_fixed_sites()
        self._fixed_bases_len = sum([r.min_len for r, _ in self._fixed_regions])

    def _get_fixed_sites(self) -> list[tuple[Region, tuple[int, int]]]:
        fixed_regions: list[tuple[Region, tuple[int, int]]] = []
        for r in self._assay.get_regions_by_sequence_type(SequenceType.FIXED):
            pos = get_position_in_parent(self._assay, r.region_id)
            fixed_regions.append((r, pos))

        return fixed_regions

    def descriptive_identifier(self) -> str:
        """Return a descriptive identifier for reporting.

        See `HasFilterStatistics`.
        """
        return "invalid_amplicon"

    def get_statistics_name(self):
        """Return a descriptive identifier for detailed statistics.

        See `HasCustomStatistics`.
        """
        return "amplicon"

    def get_statistics(self) -> AmpliconBuilderStatistics:
        """Return the statistics for the amplicon builder.

        See `HasCustomStatistics`.
        """
        return self._custom_stats

    def filtered(self) -> int:
        """Return the number of reads that have been filtered by this stage."""
        return self._custom_stats.failed_reads

    def _scan_forward_read(self, read: SequenceRecord) -> AmpliconRegionSlices:
        """Scan the forward read for the different regions of the amplicon.

        :param read: the forward read
        :return: the slices for the different regions of the amplicon in the read
        """
        region_slices = AmpliconRegionSlices()

        lbs1_alm = self._lbs1_aligner.locate(read.sequence)
        lbs2_alm = None

        if lbs1_alm:
            lbs1_start_pos, lbs1_end_pos = lbs1_alm[2], lbs1_alm[3]
            region_slices.lbs1 = slice(lbs1_start_pos, lbs1_end_pos)

            if lbs1_start_pos >= self._pid_1_umi_1_region_len:
                region_slices.pid1_umi1 = slice(
                    lbs1_start_pos - self._pid_1_umi_1_region_len, lbs1_start_pos
                )
        elif len(read) >= self._pid_1_umi_1_region_len:
            # No LBS-1 found, we will assume the read is properly anchored and extract the PID-1, UMI-1 region,
            # simply from the expected position.
            region_slices.pid1_umi1 = slice(0, self._pid_1_umi_1_region_len)

        # if the read is longer than the end of the uei region and thus contains the (partial) lbs-2 region
        if len(read) > self._uei_region_end_pos:
            lbs2_alm = self._lbs2_aligner.locate(read.sequence)
            if lbs2_alm:
                # [lbs1 end pos, lbs2 start pos)
                region_slices.uei = slice(lbs1_alm[3], lbs2_alm[2])

        # Check if we have the full pbs-2, umi-2 region after lbs-2 alignment
        if lbs2_alm:
            lbs2_start_pos, lbs2_end_pos = lbs2_alm[2], lbs2_alm[3]
            region_slices.lbs2 = slice(lbs2_start_pos, lbs2_end_pos)

            # TODO: Allow partial regions here when the full LBS-2 is matched ?
            if (lbs2_end_pos + self._pid_2_umi_2_region_len) <= len(read):
                region_slices.pid2_umi2 = slice(
                    lbs2_end_pos, lbs2_end_pos + self._pid_2_umi_2_region_len
                )

        return region_slices

    def _scan_reverse_read(self, read: SequenceRecord) -> AmpliconRegionSlices:
        """Scan the reverse read for the different regions of the amplicon.

        :param read: the reverse read
        :return: the slices for the different regions of the amplicon in the read
        """
        region_slices = AmpliconRegionSlices()

        lbs2_alm = self._lbs2_rc_aligner.locate(read.sequence)
        lbs1_alm = None

        if lbs2_alm:
            lbs_2_start_pos, lbs_2_end_pos = lbs2_alm[2], lbs2_alm[3]
            region_slices.lbs2 = slice(lbs_2_start_pos, lbs_2_end_pos)
            if lbs_2_start_pos >= self._pid_2_umi_2_region_len:
                region_slices.pid2_umi2 = slice(
                    lbs_2_start_pos - self._pid_2_umi_2_region_len, lbs_2_start_pos
                )

            # if the read is longer than the end of the uei region and thus contains the (partial) lbs-2 region
            if len(read) > self._uei_region_end_pos_rev:
                lbs1_alm = self._lbs1_rc_aligner.locate(read.sequence)

                if lbs1_alm:
                    # [lbs2 end pos, lbs1 start pos)
                    region_slices.uei = slice(lbs2_alm[3], lbs1_alm[2])
        elif len(read) > self._pid_2_umi_2_region_len:
            # No LBS-2 found, we will assume the read is properly anchored and extract the PID-2, UMI-2 region,
            # simply from the expected position.
            region_slices.pid2_umi2 = slice(0, self._pid_2_umi_2_region_len)

        # Check if we have the full pbs-1, umi-1 region after lbs-2 alignment
        if lbs1_alm:
            lbs1_start_pos, lbs1_end_pos = lbs1_alm[2], lbs1_alm[3]
            region_slices.lbs1 = slice(lbs1_start_pos, lbs1_end_pos)

            # TODO: Allow partial regions here when the full LBS-1 is matched?
            if (lbs1_end_pos + self._pid_1_umi_1_region_len) <= len(read):
                region_slices.pid1_umi1 = slice(
                    lbs1_end_pos, lbs1_end_pos + self._pid_1_umi_1_region_len
                )

        return region_slices

    @staticmethod
    def _consensus_seq(
        read1: SequenceRecord, read2: SequenceRecord, region1_slice, region2_slice
    ):
        """Combine slices from forward and reverse reads into a single sequence.

        If one of the slices is None, the other slice is returned.
        If two slices are present they are combined based on the quality scores.

        :param read1: the forward read
        :param read2: the reverse read
        :param region1_slice: the slice from the forward read
        :param region2_slice: the slice from the reverse read
        """
        assert read1.qualities is not None
        assert read2.qualities is not None

        if not region1_slice and not region2_slice:
            return None, None

        if not region1_slice or not region2_slice:
            if region1_slice:
                s = read1.sequence[region1_slice]
                q = read1.qualities[region1_slice]

            else:
                s = reverse_complement(read2.sequence[region2_slice])
                q = (read2.qualities[region2_slice])[::-1]

            return s.encode("ascii"), q.encode("ascii")

        else:
            # TODO: Consensus of non LBS regions with partial lengths?
            #   We can inject both r1 and r2 into an all N template with zero qualities

            q1 = bytearray(read1.qualities[region1_slice].encode("ascii"))
            q2 = bytearray(read2.qualities[region2_slice].encode("ascii"))[::-1]

            s1 = bytearray(read1.sequence[region1_slice].encode("ascii"))
            s2 = bytearray(
                reverse_complement(read2.sequence[region2_slice]).encode("ascii")
            )

            assert len(s1) == len(s2)
            assert len(q1) == len(q2)

            s_np = np.select([q1 > q2, q1 <= q2], [s1, s2])
            q_np = np.max([q1, q2], axis=0)

            return s_np.tobytes(), q_np.tobytes()

    def _consensus_qual_lbs1(
        self, read1: SequenceRecord, read2: SequenceRecord, region1_slice, region2_slice
    ):
        """Combine the quality scores of the LBS-1 region from forward and reverse reads.

        :param read1: the forward read
        :param read2: the reverse read
        :param region1_slice: the slice from the forward read
        :param region2_slice: the slice from the reverse read
        """
        return self._consensus_qual_lbs(
            read1, read2, region1_slice, region2_slice, self._lbs1_qual
        )

    def _consensus_qual_lbs2(
        self, read1: SequenceRecord, read2: SequenceRecord, region1_slice, region2_slice
    ):
        """Combine the quality scores of the LBS-2 region from forward and reverse reads.

        :param read1: the forward read
        :param read2: the reverse read
        :param region1_slice: the slice from the forward read
        :param region2_slice: the slice from the reverse read
        """
        return self._consensus_qual_lbs(
            read1, read2, region1_slice, region2_slice, self._lbs2_qual
        )

    @staticmethod
    def _consensus_qual_lbs(
        read1: SequenceRecord,
        read2: SequenceRecord,
        region1_slice,
        region2_slice,
        template_qual,
    ) -> bytes:
        """Combine the quality scores of an LBS region from forward and reverse reads.

        :param read1: the forward read
        :param read2: the reverse read
        :param region1_slice: the slice from the forward read
        :param region2_slice: the slice from the reverse read
        :param template_qual: the quality template for the region
        """
        # We cap the length of the quality string to the length of the template
        # Due to indels the quality string can be longer than the template
        # The quality string is thus not 100% accurate in the presence of indels
        # Since we only use this for Q30 statistics, this is not a big issue
        # For fully accurate quality scores we would need to have the full alignment
        # instead of just the matched region and this has a very high overhead.
        assert read1.qualities is not None
        assert read2.qualities is not None

        if not region1_slice and not region2_slice:
            return template_qual

        if not region1_slice or not region2_slice:
            tq = bytearray(template_qual)

            if region1_slice:
                q = read1.qualities[region1_slice].encode("ascii")[: len(tq)]
                tq[: len(q)] = q

            else:
                q = (read2.qualities[region2_slice])[::-1].encode("ascii")[: len(tq)]
                tq[-len(q) :] = q

            return bytes(tq)

        else:
            tq1 = bytearray(template_qual)
            tq2 = bytearray(template_qual)

            q1 = read1.qualities[region1_slice].encode("ascii")[: len(template_qual)]
            tq1[: len(q1)] = q1

            q2 = (read2.qualities[region2_slice])[::-1].encode("ascii")[
                : len(template_qual)
            ]
            tq2[-len(q2) :] = q2

            assert len(tq1) == len(tq2)
            tq_numpy = np.max([tq1, tq2], axis=0)

            return tq_numpy.tobytes()

    def _check_regions(
        self, pid1_umi1_region: str, uei_region: str, pid2_umi2_region: str
    ) -> AmpliconBuilderFailureReason | None:
        """Check for valid regions in the read pair.

        Return True if all regions are valid, False otherwise.
        Failure counters are incremented in the statistics object.

        :param pid1_umi1_region: the PID-1 and UMI-1 region
        :param uei_region: the UEI region
        :param pid2_umi2_region: the PID-2 and UMI-2 region
        :return: None if all regions are valid, a failure reason otherwise
        """
        if not pid1_umi1_region:
            self._custom_stats.failed_missing_upi1_umi1_reads += 1
            return AmpliconBuilderFailureReason.MISSING_PID1_UMI1

        if not pid2_umi2_region:
            self._custom_stats.failed_missing_upi2_umi2_reads += 1
            return AmpliconBuilderFailureReason.MISSING_PID2_UMI2

        # We can recover from this
        if not uei_region:
            self._custom_stats.passed_missing_uei_reads += 1
            return None

        if len(pid1_umi1_region) != self._pid_1_umi_1_region_len:
            self._custom_stats.failed_partial_upi1_umi1_reads += 1
            return AmpliconBuilderFailureReason.PARTIAL_PID1_UMI1

        if len(pid2_umi2_region) != self._pid_2_umi_2_region_len:
            self._custom_stats.failed_partial_upi2_umi2_reads += 1
            return AmpliconBuilderFailureReason.PARTIAL_PID2_UMI2

        # We can recover from this
        if len(uei_region) != self._uei_region_len:
            self._custom_stats.passed_partial_uei_reads += 1
            return None

        return None

    def __call__(
        self,
        read1: SequenceRecord,
        read2: SequenceRecord,
        info1: ModificationInfo,
        info2: ModificationInfo,
    ) -> Optional[SequenceRecord]:
        """Build an amplicon from paired-end reads (read1, read2).

        Return the processed read pair or None if the read pair has been
        "consumed" (filtered or written to an output file)
        and should thus not be passed on to subsequent steps.

        :param read1: the forward read
        :param read2: the reverse read
        :param info1: the modification info for the forward read, (ignored)
        :param info2: the modification info for the reverse read, (ignored)
        :return: the created amplicon as a new SequenceRecord or None if filtered out
        """
        # Create a slice for each region in the amplicon (if it could be found)
        r1_regions = self._scan_forward_read(read1)
        r2_regions = self._scan_reverse_read(read2)

        # Combine the info from forward and reverse reads
        pid1_umi1_region_seq, pid1_umi1_region_qual = self._consensus_seq(
            read1, read2, r1_regions.pid1_umi1, r2_regions.pid1_umi1
        )
        pid2_umi2_region_seq, pid2_umi2_region_qual = self._consensus_seq(
            read1, read2, r1_regions.pid2_umi2, r2_regions.pid2_umi2
        )
        uei_region_seq, uei_region_qual = self._consensus_seq(
            read1, read2, r1_regions.uei, r2_regions.uei
        )
        lbs1_region_qual = self._consensus_qual_lbs1(
            read1, read2, r1_regions.lbs1, r2_regions.lbs1
        )
        lbs2_region_qual = self._consensus_qual_lbs2(
            read1, read2, r1_regions.lbs2, r2_regions.lbs2
        )

        # Check for errors and increment the statistics
        error = self._check_regions(
            pid1_umi1_region_seq, uei_region_seq, pid2_umi2_region_seq
        )

        # Write failed reads to the "failed" writer
        if error is not None:
            self._custom_stats.failed_reads += 1
            if self._writer is not None:
                read1.name += f" {error.value}"
                read2.name += f" {error.value}"
                # Add an error message to the header
                self._writer.write(read1, read2)
            return None

        if r1_regions.lbs1 is None:
            self._custom_stats.passed_missing_lbs1_anchor += 1

        # Combine the regions into a single amplicon sequence
        seq = self._region_combiner.build_sequence(
            pid1_umi1_region_seq,
            pid2_umi2_region_seq,
            uei_region_seq,
        )
        qual = self._region_combiner.build_qualities(
            pid1_umi1_region_qual,
            pid2_umi2_region_qual,
            uei_region_qual,
            lbs1_region_qual,
            lbs2_region_qual,
        )

        assert len(seq) == len(qual)

        # Create a new amplicon template to fill with the reads
        amplicon = SequenceRecord(
            name=read1.name,
            sequence=seq.decode("ascii"),
            qualities=qual.decode("ascii"),
        )

        self._stats.update(amplicon.sequence)
        self._custom_stats.passed_reads += 1

        return amplicon
