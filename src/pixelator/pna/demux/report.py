"""Routines for printing a report.

Copyright © 2024 Pixelgen Technologies AB.
"""

from typing import Any, Dict

import pydantic
from cutadapt.report import Statistics

from pixelator.pna.read_processing.statistics import HasCustomStatistics
from pixelator.pna.report.models import SampleReport


class BarcodeCorrectionStatistics(Statistics):
    """Collect statistics for the barcode correction step."""

    FILTERS = {"invalid_antibody_barcode": "no valid antibody barcodes found"}

    def __init__(self) -> None:
        """Initialize the BarcodeCorrectionStatistics collector."""
        super().__init__()
        self.paired = False
        self._custom_stats: dict[str, Any] = dict()

    def __iadd__(self, other: Any):
        """Merge statistics from another object into this one."""
        super().__iadd__(other)
        if hasattr(other, "_custom_stats"):
            for name, value in other._custom_stats.items():
                if name in self._custom_stats:
                    self._custom_stats[name] += value
                else:
                    self._custom_stats[name] = value

        return self

    @property
    def pid_pair_counter(self):
        """Return the number of reads per PID pair."""
        return self._custom_stats["demux"].get_pid_group_counter()

    def as_json(self, gc_content: float = 0.5, one_line: bool = False) -> Dict:
        """Return a dict representation suitable for dumping in JSON format.

        To achieve a more compact representation, set one_line to True, which
        will wrap some items in a `cutadapt.json.OneLine` object, and use
        `cutadapt.json.dumps` instead of `json.dumps` to dump the dict.
        """
        demux_report = None

        assert "demux" in self._custom_stats
        demux_report = self._custom_stats["demux"].collect()
        return demux_report

    def _collect_step(self, step):
        super()._collect_step(step)
        if isinstance(step, HasCustomStatistics):
            name = step.get_statistics_name()
            if name in self._custom_stats:
                self._custom_stats[name] += step.get_statistics()
            else:
                self._custom_stats[name] = step.get_statistics()

    def collect(
        self,
        n: int,
        total_bp1: int,
        total_bp2: int | None,
        modifiers,
        steps,
        set_paired_to_none: bool = False,
    ):
        """Enable stats.paired to be set to None when unknown."""
        stats = super().collect(n, total_bp1, total_bp2, modifiers, steps)
        if set_paired_to_none:
            stats.paired = None
        return stats


class DemuxSampleReport(SampleReport):
    """Model for a demux sample report."""

    input_reads: int = pydantic.Field(
        ..., description="The number of input reads processed."
    )
    output_reads: int = pydantic.Field(
        ...,
        description="The number of reads that have valid marker barcodes and are demuxed.\nThis corresponds to passed_exact_reads + passed_corrected_reads.",
    )
    failed_reads: int = pydantic.Field(
        ...,
        description="The number of reads with barcodes that could not be corrected against the panel.",
    )
    output_exact_reads: int = pydantic.Field(
        ...,
        description="The number of reads for which the antibody barcodes were an exact match with the panel.",
    )
    output_corrected_reads: int = pydantic.Field(
        ...,
        description="The number of reads for which the antibody barcodes needed error correction before matching with the panel.",
    )
    invalid_pid1_reads: int = pydantic.Field(
        ...,
        description="The number of amplicon reads that where discarded because of an unrecognised PID1 region.",
    )
    invalid_pid2_reads: int = pydantic.Field(
        ...,
        description="The number of amplicon reads that where discarded because of an unrecognised PID2 region.",
    )
    invalid_pid1_pid2_reads: int = pydantic.Field(
        ...,
        description="The number of amplicon reads that where discarded because of both the PID1 and PID2 regions being unrecognised.",
    )
    pid1_matches_distance_distribution: dict[int, int] = pydantic.Field(
        ..., description="The distribution of distances for PID1 matches."
    )
    pid2_matches_distance_distribution: dict[int, int] = pydantic.Field(
        ..., description="The distribution of distances for PID2 matches."
    )
    pid_group_sizes: list[tuple[str, str, int]] = pydantic.Field(
        ..., description="The number of reads per PID pair."
    )

    @pydantic.computed_field(  # type: ignore
        description="The fraction of reads that were discarded by the demux step.",
        return_type=float,
    )
    @property
    def fraction_failed_reads(self) -> float:
        """Calculate the fraction of reads that were discarded by the demux step."""
        if self.input_reads == 0:
            return 0
        return self.failed_reads / self.input_reads
