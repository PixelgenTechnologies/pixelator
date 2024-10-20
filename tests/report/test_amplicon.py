"""Tests for PixelatorReporting related to the amplicon stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pixelator.report import PixelatorReporting, PixelatorWorkdir
from pixelator.report.models.amplicon import AmpliconSampleReport


@pytest.fixture()
def amplicon_summary_input(
    pixelator_workdir, amplicon_stage_all_reports
) -> PixelatorWorkdir:
    return pixelator_workdir


expected = [
    AmpliconSampleReport(
        sample_id="pbmcs_unstimulated",
        fraction_q30=0.9606927317923855,
        fraction_q30_bc=0.9660959533395918,
        fraction_q30_pbs1=0.9683133231643615,
        fraction_q30_pbs2=0.9477632357943524,
        fraction_q30_umi=0.965813230797693,
        fraction_q30_upia=0.9602759816852228,
        fraction_q30_upib=0.9671804579646854,
    ),
    AmpliconSampleReport(
        sample_id="uropod_control",
        fraction_q30=0.9602150898560277,
        fraction_q30_bc=0.9624244207098193,
        fraction_q30_pbs1=0.9721885264356955,
        fraction_q30_pbs2=0.9535804849402897,
        fraction_q30_umi=0.9627836138165338,
        fraction_q30_upia=0.9523661337597942,
        fraction_q30_upib=0.9642700474644683,
    ),
]


@pytest.mark.parametrize("sample_name,expected", [(r.sample_id, r) for r in expected])
def test_adapterqc_metrics_lookup(amplicon_summary_input, sample_name, expected):
    reporting = PixelatorReporting(amplicon_summary_input)
    r = reporting.amplicon_metrics(sample_name)
    assert r == expected


def test_amplicon_summary(amplicon_summary_input, snapshot):
    reporting = PixelatorReporting(amplicon_summary_input)
    result = reporting.amplicon_summary()

    snapshot.assert_match(result.to_csv(), "amplicon_summary.csv")
