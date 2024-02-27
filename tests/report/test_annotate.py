"""Tests for PixelatorReporting related to the annotate stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models import SummaryStatistics
from pixelator.report.models.annotate import AnnotateSampleReport


@pytest.fixture()
def annotate_summary_input(
    pixelator_workdir, annotate_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_annotate_metrics_lookup(annotate_summary_input, sample_name, snapshot):
    reporting = PixelatorReporting(annotate_summary_input)
    result = reporting.annotate_metrics(sample_name)

    snapshot.assert_match(
        result.to_json(indent=4), f"{sample_name}_annotate_metrics.json"
    )


def test_annotate_summary(annotate_summary_input, snapshot):
    reporting = PixelatorReporting(annotate_summary_input)
    result = reporting.annotate_summary()

    snapshot.assert_match(result.to_csv(), "annotate_summary.csv")


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_annotate_no_aggregate_recovery_computed_fields(
    annotate_summary_input, sample_name
):
    """Test that computed fields return None when aggregate recovery is disabled."""

    reporting = PixelatorReporting(annotate_summary_input)
    result = reporting.annotate_metrics(sample_name)
    result = result.model_copy(
        update={
            "aggregate_count": None,
            "molecules_in_aggregates_count": None,
            "reads_in_aggregates_count": None,
        }
    )

    assert result.aggregate_count is None
    assert result.molecules_in_aggregates_count is None
    assert result.reads_in_aggregates_count is None
    assert result.fraction_reads_in_aggregates is None
    assert result.fraction_molecules_in_aggregates is None
