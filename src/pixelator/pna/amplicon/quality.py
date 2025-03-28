"""Q30 quality statistics for the amplicon regions.

Copyright © 2024 Pixelgen Technologies AB.
"""

import typing
from collections import Counter, defaultdict

import numpy as np
from cutadapt.info import ModificationInfo
from cutadapt.steps import SingleEndStep
from dnaio import SequenceRecord

from pixelator.pna.config import PNAAssay
from pixelator.pna.config.assay import get_position_in_parent
from pixelator.pna.read_processing.statistics import HasCustomStatistics


class QualityStatisticsResult(typing.TypedDict):
    """The result of the quality statistics."""

    fraction_q30_total: float
    fraction_q30_umi1: float
    fraction_q30_pid1: float
    fraction_q30_lbs1: float
    fraction_q30_uei: float
    fraction_q30_lbs2: float
    fraction_q30_pid2: float
    fraction_q30_umi2: float


class QualityStatistics:
    """A pipline step that calculates the Q30 fraction for each region in the amplicon."""

    def __init__(self, data):
        """Initialize the QualityStatistics object.

        :param data: A dictionary with the quality statistics for each region.
        """
        self._region_counters = {}
        for region_id, region_data in data.items():
            self._region_counters[region_id] = Counter(region_data)

    def total_bases(self, region_id: str):
        """Return the total number of bases for a region."""
        return self._region_counters[region_id]["total_bases"]

    def sequences_bases(self, region_id: str):
        """Return the number of sequenced bases for a region."""
        return self._region_counters[region_id]["sequenced_bases"]

    def q30_bases(self, region_id: str):
        """Return the number of bases with quality above 30 for a region."""
        return self._region_counters[region_id]["q30_bases"]

    def get_q30_fraction_total_bases(self, region_id: str):
        """Return the fraction of bases with quality above 30 for a region.

        The fraction is calculated as the number of bases with quality above
        30 divided by the total number bases in the region, even those that were not sequenced.
        """
        r = self._region_counters[region_id]
        if (bases := r["total_bases"]) == 0:
            return 0

        return r["q30_bases"] / bases

    def get_q30_fraction(self, region_id: str):
        """Return the fraction of bases with quality above 30 for a region.

        The fraction is calculated as the number of bases with quality above
        30 divided by the total number of sequenced bases.
        """
        r = self._region_counters[region_id]
        if (bases := r["sequenced_bases"]) == 0:
            return 0

        return r["q30_bases"] / bases

    def __iadd__(self, other):
        """Merge statistics from another object into this one."""
        for region_id, region_data in other._region_counters.items():
            if region_id not in self._region_counters:
                self._region_counters[region_id] = other._region_counters[region_id]
            else:
                self._region_counters[region_id] += other._region_counters[region_id]

        return self

    def collect(self) -> QualityStatisticsResult:
        """Return the quality statistics for the complete amplicon and all regions."""
        return QualityStatisticsResult(
            fraction_q30_total=self.get_q30_fraction("amplicon"),
            fraction_q30_umi1=self.get_q30_fraction("umi-1"),
            fraction_q30_pid1=self.get_q30_fraction("pid-1"),
            fraction_q30_lbs1=self.get_q30_fraction("lbs-1"),
            fraction_q30_uei=self.get_q30_fraction_total_bases("uei"),
            fraction_q30_lbs2=self.get_q30_fraction("lbs-2"),
            fraction_q30_pid2=self.get_q30_fraction("pid-2"),
            fraction_q30_umi2=self.get_q30_fraction("umi-2"),
        )


class QualityProfileStep(SingleEndStep, HasCustomStatistics):
    """A pipeline step that calculates the Q30 fraction for each region in the amplicon."""

    def __init__(self, assay: PNAAssay):
        """Initialize the QualityProfileStep object.

        :param assay: The assay object
        """
        super().__init__()
        self.assay = assay
        self._amplicon_len = assay.get_region_by_id("amplicon").min_len
        self._total_bases = np.zeros(self._amplicon_len, dtype=np.uint64)
        self._total_sequenced_bases = np.zeros(self._amplicon_len, dtype=np.uint64)
        self._q30_bases = np.zeros(self._amplicon_len, dtype=np.uint64)

        self._region_slices = self._get_region_slices()

    def _get_region_slices(self):
        """Return the [start, stop) positions for each region in the amplicon."""
        amplicon_region = self.assay.get_region_by_id("amplicon")
        region_slices = {}

        if amplicon_region is None:
            raise ValueError("Assay does not contain an amplicon region")

        for r in amplicon_region.get_leaves():
            region_slices[r.region_id] = slice(
                *get_position_in_parent(self.assay, r.region_id)
            )

        return region_slices

    def __call__(self, read: SequenceRecord, info: ModificationInfo) -> SequenceRecord:
        """Create a quality profile on each read position for the amplicon."""
        # Create a view into the original quality array
        qual = np.frombuffer(read.qualities_as_bytes(), dtype=np.int8)
        # Convert the vector to integers
        int_qual = qual - 33

        # Increment the total number of bases by one
        # Skip the bases with quality less than 2.
        # Those are mostly LBS region bases that were not sequenced but known from the assay design.
        self._total_bases += 1
        self._total_sequenced_bases += int_qual > 2
        # # Increment the number of bases with quality above 30 by one
        self._q30_bases += int_qual > 30
        return read

    def get_statistics_name(self) -> str:
        """Return the name of the statistics."""
        return "quality_profile"

    def get_statistics(self) -> QualityStatistics:
        """Return the quality statistics."""
        stats: dict[str, dict[str, int]] = defaultdict(dict)

        stats["amplicon"]["q30_bases"] = np.sum(self._q30_bases, dtype=int)
        stats["amplicon"]["sequenced_bases"] = np.sum(
            self._total_sequenced_bases, dtype=int
        )
        stats["amplicon"]["total_bases"] = np.sum(self._total_bases, dtype=int)

        for region_id, region_slice in self._region_slices.items():
            stats[region_id]["q30_bases"] = np.sum(
                self._q30_bases[region_slice], dtype=int
            )
            stats[region_id]["sequenced_bases"] = np.sum(
                self._total_sequenced_bases[region_slice], dtype=int
            )
            stats[region_id]["total_bases"] = np.sum(
                self._total_bases[region_slice], dtype=int
            )

        return QualityStatistics(stats)
