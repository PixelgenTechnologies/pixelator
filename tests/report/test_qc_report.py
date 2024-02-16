"""Test QC report generation.

Copyright © 2022 Pixelgen Technologies AB.
"""

import subprocess
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from pixelator.report.qcreport import make_report


@pytest.fixture(scope="session")
def full_run_assets_dir(request) -> Path:
    subprocess.run(
        "task tests:update-web-test-data",
        shell=True,
        cwd=str(request.config.rootdir),
        capture_output=True,
        check=True,
    )
    return Path(__file__).parent / "assets/full_run"


# TODO: Needs full data. Currently, the test is failing because of missing data.


@pytest.mark.web_test
@pytest.mark.parametrize("report_file", ("uropod_control_300k_S1_001.qc-report.html",))
def test_make_report(full_run_assets_dir, tmp_path, page: Page, report_file: str):
    """Use playwright to render the QC report and check if all sections are present."""
    make_report(
        full_run_assets_dir,
        tmp_path,
        panel="human-sc-immunology-spatial-proteomics",
        metadata=None,
        verbose=True,
    )

    # assert that the report was created
    assert (tmp_path / report_file).exists()

    page.goto(f"file://{str(tmp_path / report_file)}")

    parameter_section = page.locator('//span[text()="Parameters"]')
    expect(parameter_section).to_have_text("Parameters")

    cells_section = page.locator('//span[text()="Cells"]')
    expect(cells_section).to_have_text("Cells")

    annotation_section = page.locator('//span[text()="Cell annotations"]')
    expect(annotation_section).to_have_text("Cell annotations")

    antibody_section = page.locator('//span[text()="Antibodies"]')
    expect(antibody_section).to_have_text("Antibodies")

    sequencing_section = page.locator('//span[text()="Sequencing"]')
    expect(sequencing_section).to_have_text("Sequencing")

    # Maybe just checking the presence of each metric is useful to
