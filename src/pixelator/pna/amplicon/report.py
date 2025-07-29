"""Routines for printing a report.

Copyright © 2024 Pixelgen Technologies AB.
"""

import typing
from typing import Any, Dict, Optional

import pydantic
from cutadapt.report import Statistics

from pixelator.pna.read_processing.statistics import HasCustomStatistics
from pixelator.pna.report.models.base import SampleReport


class Q30Statistics(pydantic.BaseModel):
    """The result of the quality statistics."""

    total: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the complete amplicon.",
    )
    umi1: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UMI1 region.",
    )
    pid1: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the PID1 region.",
    )
    lbs1: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the LBS1 region.",
    )
    uei: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UEI region.",
    )
    lbs2: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the LBS2 region.",
    )
    pid2: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the PID2 region.",
    )
    umi2: float = pydantic.Field(
        ...,
        description="The fraction of reads with Phred score ≥ 30 in the UMI2 region.",
    )


class BasesCountStatistics(pydantic.BaseModel):
    """The result of the base count statistics."""

    input: int
    input_read1: int = pydantic.Field(
        ..., description="The number of input bases in read1."
    )
    input_read2: int = pydantic.Field(
        ..., description="The number of input bases in read2."
    )
    quality_trimmed: int = pydantic.Field(
        ..., description="The total number of quality trimmed bases."
    )
    quality_trimmed_read1: int = pydantic.Field(
        ..., description="The number of quality trimmed bases in read1."
    )
    quality_trimmed_read2: Optional[int] = pydantic.Field(
        ..., description="The number of quality trimmed bases in read2."
    )
    output: int = pydantic.Field(..., description="The total number of output bases.")


class AmpliconSampleReport(SampleReport):
    """Model for an amplicon sample report."""

    report_type: typing.Literal["amplicon"] = "amplicon"

    input_reads: int = pydantic.Field(
        ..., description="The total number of input reads in the amplicon stage."
    )

    output_reads: int = pydantic.Field(
        ...,
        description="The number of reads that passed the filters in the amplicon stage.",
    )

    passed_missing_uei_reads: int = pydantic.Field(
        ...,
        description="The number of reads that passed but do not have a recognisable UEI region.",
    )

    passed_partial_uei_reads: int = pydantic.Field(
        ...,
        description="The number of reads that passed but do not have a fully sequenced UEI region.",
    )

    passed_missing_lbs1_anchor: int = pydantic.Field(
        ...,
        description="The number of reads that passed but do not have a recognisable LBS1 anchor.",
    )

    failed_too_many_n_reads: int = pydantic.Field(
        ...,
        description="The number of reads discarded because they contain too many ambiguous bases.",
    )

    @pydantic.computed_field(
        return_type=int,
        description=(
            "The number of reads discarded because they are missing or have a partial region."
            "This is the sum of the failed or partial upi1/umi1 and upi2/umi2 reads."
        ),
    )
    def failed_invalid_amplicon_reads(self) -> int:
        """Calculate the number of reads discarded because they are missing or have a partial region."""
        return (
            self.failed_partial_upi1_umi1_reads
            + self.failed_partial_upi2_umi2_reads
            + self.failed_missing_upi1_umi1_reads
            + self.failed_missing_upi2_umi2_reads
        )

    failed_partial_upi1_umi1_reads: int = pydantic.Field(
        ...,
        description="The number of reads discarded because they have a partial UPI1/UMI1 region.",
    )

    failed_partial_upi2_umi2_reads: int = pydantic.Field(
        ...,
        description="The number of reads discarded because they have a partial UPI2/UMI2 region.",
    )

    failed_missing_upi1_umi1_reads: int = pydantic.Field(
        ...,
        description="The number of reads discarded because they are missing UPI1/UMI1 sequences.",
    )

    failed_missing_upi2_umi2_reads: int = pydantic.Field(
        ...,
        description="The number of reads discarded because they are missing UPI2/UMI2 sequences.",
    )

    total_failed_reads: int = pydantic.Field(
        ...,
        description="The total number of reads that failed the filters and are discarded.",
    )

    q30_statistics: Q30Statistics = pydantic.Field(
        ..., description="Q30 statistics for the amplicon stage."
    )

    basepair_counts: BasesCountStatistics = pydantic.Field(
        ..., description="Base count statistics for the amplicon stage."
    )

    @pydantic.computed_field(  # type: ignore
        description="The fraction of reads that was discarded by the amplicon step.",
        return_type=float,
    )
    @property
    def fraction_discarded_reads(self) -> float:
        """Calculate the fraction of reads that were discarded by the amplicon step."""
        if self.input_reads == 0:
            return 0
        return self.total_failed_reads / self.input_reads


class AmpliconStatistics(Statistics):
    """Statistics for the amplicon stage."""

    FILTERS = {
        "too_many_n": "with too many N",
        "invalid_amplicon": "with missing amplicon regions",
    }

    def __init__(self) -> None:
        """Initialize the AmpliconStatistics object."""
        super().__init__()
        self.paired: bool | None = None
        self._custom_stats: dict[str, Any] = dict()

    def __iadd__(self, other: Any):
        """Merge statistics from another object into this one."""
        if other.paired is None:
            other.paired = self.paired
        super().__iadd__(other)
        if hasattr(other, "_custom_stats"):
            for name, value in other._custom_stats.items():
                if name in self._custom_stats:
                    self._custom_stats[name] += value
                else:
                    self._custom_stats[name] = value

        return self

    def as_json(self, gc_content: float = 0.5, one_line: bool = False) -> Dict:
        """Return a dict representation suitable for dumping in JSON format."""
        return self.as_dict()

    def as_dict(self) -> Dict:
        """Return a dict representation of the class."""
        filtered = {name: self.filtered.get(name) for name in self.FILTERS.keys()}
        filtered_total = sum(self.filtered.values())
        written_reads = self.read_length_statistics.written_reads()
        written_bp = self.read_length_statistics.written_bp()
        assert (written_reads + filtered_total) == self.n

        q30_regions_report = None
        amplicon_report = None
        if "quality_profile" in self._custom_stats:
            q30_regions_report = self._custom_stats["quality_profile"].collect()

        if "amplicon" in self._custom_stats:
            amplicon_report = self._custom_stats["amplicon"].collect()

        # type: ignore
        q30_stats = Q30Statistics(
            total=q30_regions_report["fraction_q30_total"],  # type: ignore
            umi1=q30_regions_report["fraction_q30_umi1"],  # type: ignore
            pid1=q30_regions_report["fraction_q30_pid1"],  # type: ignore
            lbs1=q30_regions_report["fraction_q30_lbs1"],  # type: ignore
            uei=q30_regions_report["fraction_q30_uei"],  # type: ignore
            lbs2=q30_regions_report["fraction_q30_lbs2"],  # type: ignore
            pid2=q30_regions_report["fraction_q30_pid2"],  # type: ignore
            umi2=q30_regions_report["fraction_q30_umi2"],  # type: ignore
        )
        basepair_counts = BasesCountStatistics(
            input=self.total,  # type: ignore
            input_read1=self.total_bp[0],  # type: ignore
            input_read2=self.total_bp[1],  # type: ignore
            quality_trimmed=self.quality_trimmed,  # type: ignore
            quality_trimmed_read1=self.quality_trimmed_bp[0],  # type: ignore
            quality_trimmed_read2=self.quality_trimmed_bp[1],  # type: ignore
            output=self.total_written_bp,  # type: ignore
        )

        # type: ignore
        report = AmpliconSampleReport(
            sample_id="",
            report_type="amplicon",
            product_id="single-cell-pna",
            input_reads=self.n,
            output_reads=written_reads,
            total_failed_reads=filtered_total,
            passed_missing_uei_reads=amplicon_report["passed_missing_eui_reads"],  # type: ignore
            passed_partial_uei_reads=amplicon_report["passed_partial_uei_reads"],  # type: ignore
            passed_missing_lbs1_anchor=amplicon_report["passed_missing_lbs1_anchor"],  # type: ignore
            failed_too_many_n_reads=self.filtered["too_many_n"],  # type: ignore
            failed_partial_upi1_umi1_reads=amplicon_report[
                "failed_partial_upi1_umi1_reads"
            ],  # type: ignore
            failed_partial_upi2_umi2_reads=amplicon_report[
                "failed_partial_upi2_umi2_reads"
            ],  # type: ignore
            failed_missing_upi1_umi1_reads=amplicon_report[
                "failed_missing_upi1_umi1_reads"
            ],  # type: ignore
            failed_missing_upi2_umi2_reads=amplicon_report[
                "failed_missing_upi2_umi2_reads"
            ],  # type: ignore
            q30_statistics=q30_stats,
            basepair_counts=basepair_counts,
        )
        return report.model_dump(
            mode="python", exclude={"sample_id", "report_type", "product_id"}
        )

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
