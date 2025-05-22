"""Collect statistics for amplicon.

Copyright © 2023 Pixelgen Technologies AB.
"""

import collections
import dataclasses
from functools import cache
from typing import Any, Union

import numba as nb
import numpy as np

from pixelator.common.config import RegionType, get_position_in_parent
from pixelator.mpx.config import config


@nb.njit
def _count_elem_in_array_where_greater_than(arr, value):
    result = 0
    for x in arr:
        if x > value:
            result += 1
    return result


@nb.njit
def _count_elem_in_array_where_greater_or_equal_than(arr, value):
    result = 0
    for x in arr:
        if x >= value:
            result += 1
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class SequenceQualityStats:
    """Container for sequence quality statistics."""

    fraction_q30_upia: float
    fraction_q30_upib: float
    fraction_q30_umi: float
    fraction_q30_pbs1: float
    fraction_q30_pbs2: float
    fraction_q30_bc: float
    fraction_q30: float

    def asdict(self) -> dict[str, Any]:
        """Return a dictionary representation of this instance."""
        return {k: v for k, v in dataclasses.asdict(self).items()}


class SequenceQualityStatsCollector:
    """Accumulate read quality statistics for a given design."""

    def __init__(self, design_name: str):
        """Accumulate read quality statistics for a given design.

        :param design_name: The name of the design of the reads for which to statistics.
        """
        design = config.get_assay(design_name)

        if design is None:
            raise ValueError(f"Unknown design: {design_name}")

        self.design = design
        self._counter: collections.Counter[str] = collections.Counter()
        self._positions = {}

        self._amplicon_len = len(design)

        amplicon = self.design.get_region_by_id("amplicon")
        if amplicon is None:
            raise ValueError("Assay does not contain an amplicon region")

        for leave in amplicon.get_leaves():
            self._positions[leave.region_id] = get_position_in_parent(
                self.design, leave.region_id
            )

        # Verify all the right regions are present
        # Check for upi, pbs and bc regions
        for region_id in ("upi-a", "upi-b", "pbs-1", "pbs-2", "bc"):
            if region_id not in self._positions:
                raise ValueError(f"Assay does not contain region {region_id}")

        # Check that at least one umi region is present
        umi_regions = self.design.get_regions_by_type(RegionType.UMI)
        if not umi_regions:
            raise ValueError("Assay does not contain a UMI region")

    @cache
    def get_position(self, region_id: str) -> tuple[int, int]:
        """Return the positions for a region.

        :param region_id: id of the region
        :returns: a tuple with start and end positions
        :raise ValueError: An unknown region id was given
        """
        r = self._positions.get(region_id)
        if r is None:
            raise ValueError(f"Unknown region: {region_id}")
        return r

    @staticmethod
    def _read_stats(quali: np.ndarray) -> tuple[int, int]:
        bases_in_read = _count_elem_in_array_where_greater_than(quali, 2)
        q30_bases_in_read = _count_elem_in_array_where_greater_or_equal_than(quali, 30)
        return bases_in_read, q30_bases_in_read

    def _umi_stats(self, quali: np.ndarray) -> tuple[int, int]:
        # Q30 Bases in UMI
        umi_regions = self.design.get_regions_by_type(RegionType.UMI)
        umi_positions = [self.get_position(r.region_id) for r in umi_regions]

        p = np.zeros(self._amplicon_len, dtype=np.uint8)
        for pos in umi_positions:
            p[slice(*pos)] = 1

        q = p * quali

        bases_in_umi = _count_elem_in_array_where_greater_than(q, 2)
        q30_bases_in_umi = _count_elem_in_array_where_greater_or_equal_than(q, 30)

        return bases_in_umi, q30_bases_in_umi

    def _get_stats_from_position(self, quali: np.ndarray, pos: str) -> tuple[int, int]:
        upia_pos = self.get_position(pos)
        slice_obj = slice(*upia_pos)
        quali_subset = quali[slice_obj]
        bases = _count_elem_in_array_where_greater_than(quali_subset, 2)
        q30 = _count_elem_in_array_where_greater_or_equal_than(quali_subset, 30)
        return bases, q30

    def _upia_stats(self, quali: np.ndarray) -> tuple[int, int]:
        return self._get_stats_from_position(quali, "upi-a")

    def _upib_stats(self, quali: np.ndarray) -> tuple[int, int]:
        return self._get_stats_from_position(quali, "upi-b")

    def _pbs1_stats(self, quali: np.ndarray) -> tuple[int, int]:
        return self._get_stats_from_position(quali, "pbs-1")

    def _pbs2_stats(self, quali: np.ndarray) -> tuple[int, int]:
        return self._get_stats_from_position(quali, "pbs-2")

    def _bc_stats(self, quali: np.ndarray) -> tuple[int, int]:
        return self._get_stats_from_position(quali, "bc")

    @property
    def read_count(self) -> int:
        """Return the number of reads processed."""
        return self._counter["read_count"]

    @property
    def stats(self) -> SequenceQualityStats:
        """Return the accumulated statistics as a SequenceQualityStats object."""
        fraction_q30_upia = (
            self._counter["q30_bases_in_upia"] / self._counter["bases_in_upia"]
        )
        fraction_q30_upib = (
            self._counter["q30_bases_in_upib"] / self._counter["bases_in_upib"]
        )
        fraction_q30_umi = (
            self._counter["q30_bases_in_umi"] / self._counter["bases_in_umi"]
        )
        fraction_q30_pbs1 = (
            self._counter["q30_bases_in_pbs1"] / self._counter["bases_in_pbs1"]
        )
        fraction_q30_pbs2 = (
            self._counter["q30_bases_in_pbs2"] / self._counter["bases_in_pbs2"]
        )
        fraction_q30_bc = (
            self._counter["q30_bases_in_bc"] / self._counter["bases_in_bc"]
        )
        fraction_q30 = (
            self._counter["q30_bases_in_read"] / self._counter["bases_in_read"]
        )

        return SequenceQualityStats(
            fraction_q30_upia=fraction_q30_upia,
            fraction_q30_upib=fraction_q30_upib,
            fraction_q30_umi=fraction_q30_umi,
            fraction_q30_pbs1=fraction_q30_pbs1,
            fraction_q30_pbs2=fraction_q30_pbs2,
            fraction_q30_bc=fraction_q30_bc,
            fraction_q30=fraction_q30,
        )

    def update(self, qualities: Union[str, np.ndarray]) -> None:
        """Update the statistics with the given read qualities."""
        # Use numpy for vectorized operations
        # Reinterpret cast to integers (same as ord)
        if isinstance(qualities, str):
            quali = np.frombuffer(qualities.encode(), dtype=np.uint8) - 33
        else:
            quali = qualities

        bases_in_read, q30_bases_in_read = self._read_stats(quali)
        bases_in_umi, q30_bases_in_umi = self._umi_stats(quali)
        bases_in_upia, q30_bases_in_upia = self._upia_stats(quali)
        bases_in_upib, q30_bases_in_upib = self._upib_stats(quali)
        bases_in_pbs1, q30_bases_in_pbs1 = self._pbs1_stats(quali)
        bases_in_pbs2, q30_bases_in_pbs2 = self._pbs2_stats(quali)
        bases_in_bc, q30_bases_in_bc = self._bc_stats(quali)

        self._counter.update(
            read_count=1,
            bases_in_read=bases_in_read,
            q30_bases_in_read=q30_bases_in_read,
            bases_in_umi=bases_in_umi,
            q30_bases_in_umi=q30_bases_in_umi,
            bases_in_upia=bases_in_upia,
            q30_bases_in_upia=q30_bases_in_upia,
            bases_in_upib=bases_in_upib,
            q30_bases_in_upib=q30_bases_in_upib,
            bases_in_pbs1=bases_in_pbs1,
            q30_bases_in_pbs1=q30_bases_in_pbs1,
            bases_in_pbs2=bases_in_pbs2,
            q30_bases_in_pbs2=q30_bases_in_pbs2,
            bases_in_bc=bases_in_bc,
            q30_bases_in_bc=q30_bases_in_bc,
        )
