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

    snapshot.assert_match(result.to_json(indent=4), "annotate_summary.json")


def test_annotate_no_aggregate_recovery_computed_fields():
    """Test that computed fields return None when aggregate recovery is disabled."""
    report = AnnotateSampleReport(
        sample_id="pbmcs_unstimulated",
        components_modularity=0.9996688156200267,
        fraction_molecules_in_largest_component=0.0006480881399870382,
        fraction_pixels_in_largest_component=0.0004888381945576015,
        input_cell_count=3052,
        input_read_count=6237,
        cell_count=3052,
        marker_count=68,
        total_marker_count=80,
        molecule_count=3086,
        a_pixel_count=3077,
        b_pixel_count=3060,
        read_count=6237,
        molecule_count_per_cell_stats=SummaryStatistics(
            mean=1.011140235910878,
            std=0.10495775843037089,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3052,
        ),
        read_count_per_cell_stats=SummaryStatistics(
            mean=2.043577981651376,
            std=0.2592989644447358,
            min=2.0,
            q1=2.0,
            q2=2.0,
            q3=2.0,
            max=5.0,
            count=3052,
        ),
        a_pixel_count_per_cell_stats=SummaryStatistics(
            mean=1.0081913499344692,
            std=0.0901346310843966,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3052,
        ),
        b_pixel_count_per_cell_stats=SummaryStatistics(
            mean=1.0026212319790302,
            std=0.05113082359929529,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3052,
        ),
        marker_count_per_cell_stats=SummaryStatistics(
            mean=1.0075360419397117,
            std=0.08648265728800528,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3052,
        ),
        a_pixel_b_pixel_ratio_per_cell_stats=SummaryStatistics(
            mean=1.0068807339449541,
            std=0.09380465569259072,
            min=0.5,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3052,
        ),
        molecule_count_per_a_pixel_per_cell_stats=SummaryStatistics(
            mean=1.002599935001625,
            std=0.050923229862335766,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3077,
        ),
        b_pixel_count_per_a_pixel_per_cell_stats=SummaryStatistics(
            mean=1.002599935001625,
            std=0.050923229862335766,
            min=1.0,
            q1=1.0,
            q2=1.0,
            q3=1.0,
            max=2.0,
            count=3077,
        ),
        aggregate_count=3052,
        molecules_in_aggregates_count=3086,
        reads_in_aggregates_count=6237,
        min_size_threshold=None,
        max_size_threshold=None,
        doublet_size_threshold=None,
        size_filter_fail_cell_count=0,
        size_filter_fail_molecule_count=0,
        size_filter_fail_read_count=0,
        pixel_count=6137,
    )

    report = report.model_copy(
        update={
            "aggregate_count": None,
            "molecules_in_aggregates_count": None,
            "reads_in_aggregates_count": None,
        }
    )

    assert report.aggregate_count is None
    assert report.molecules_in_aggregates_count is None
    assert report.reads_in_aggregates_count is None
    assert report.fraction_reads_in_aggregates is None
    assert report.fraction_molecules_in_aggregates is None
