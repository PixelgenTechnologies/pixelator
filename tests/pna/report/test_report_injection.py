"""Copyright © 2025 Pixelgen Technologies AB."""

import datetime
from pathlib import Path

import pytest

from pixelator import __version__
from pixelator.pna.report.common import PixelatorPNAReporting
from pixelator.pna.report.qcreport import SampleInfo
from pixelator.pna.report.qcreport.main import create_qc_report


@pytest.mark.slow
@pytest.mark.parametrize("sample_id", ["PNA055_Sample07_filtered_S7"])
def test_create_qc_report(
    sample_id, all_stages_all_reports_and_meta, snapshot, tmp_path
):
    reporting = PixelatorPNAReporting(all_stages_all_reports_and_meta)

    generation_time = datetime.datetime(2020, 1, 1, 0, 0, 0)

    info = SampleInfo(
        pixelator_version=__version__,
        generation_date=generation_time.isoformat(),
        sample_id=sample_id,
        sample_description="just a test",
        panel_name="human-sc-immunology-spatial-proteomics",
        panel_version="1.0.0",
        technology="PNA",
        technology_version="2.5",
        parameters=[],
    )

    r = create_qc_report(reporting, sample_id, info, tmp_path)

    report = Path(tmp_path / f"{sample_id}.qc-report.html")
    assert report.exists()
