"""Copyright © 2023 Pixelgen Technologies AB."""

import dataclasses

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.preqc import PreQCSampleReport


@pytest.fixture()
def preqc_summary_input(pixelator_workdir, preqc_stage_all_reports) -> PixelatorWorkdir:
    return pixelator_workdir


@dataclasses.dataclass
class Test:
    sample_id: str


expected = [
    PreQCSampleReport(
        sample_id="pbmcs_unstimulated",
        total_read_count=200000,
        passed_filter_read_count=199390,
        low_quality_read_count=610,
        too_many_n_read_count=0,
        too_short_read_count=0,
        too_long_read_count=0,
    ),
    PreQCSampleReport(
        sample_id="uropod_control",
        total_read_count=300000,
        passed_filter_read_count=298988,
        low_quality_read_count=1012,
        too_many_n_read_count=0,
        too_short_read_count=0,
        too_long_read_count=0,
    ),
]


@pytest.mark.parametrize("sample_name,expected", [(r.sample_id, r) for r in expected])
def test_preqc_metrics_lookup(preqc_summary_input, sample_name, expected):
    reporting = PixelatorReporting(preqc_summary_input)
    r = reporting.preqc_metrics(sample_name)
    assert r == expected


def test_preqc_summary(preqc_summary_input, snapshot):
    reporting = PixelatorReporting(preqc_summary_input)
    result = reporting.preqc_summary()

    assert_frame_equal(
        result,
        pd.DataFrame.from_records(
            data=[report.model_dump(exclude="sample_id") for report in expected],
            index=pd.Index([r.sample_id for r in expected], name="sample_id"),
        ),
    )