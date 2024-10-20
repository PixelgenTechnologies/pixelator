"""Test QC report generation.

Copyright © 2022 Pixelgen Technologies AB.
"""

import base64
import dataclasses
import gzip
import json
import re
from datetime import datetime
from pathlib import Path

import lxml
import pytest
from lxml.etree import _Element as LxmlElement
from playwright.sync_api import Page, expect

from pixelator import __version__
from pixelator.report.qcreport import (
    QCReportBuilder,
    QCReportData,
    SampleInfo,
    make_report,
)
from pixelator.report.qcreport.builder import DEFAULT_QC_REPORT_TEMPLATE
from pixelator.report.qcreport.collect import (
    collect_antibody_counts_data,
    collect_antibody_percentages_data,
    collect_component_ranked_component_size_data,
    collect_components_umap_data,
    collect_reads_per_molecule_frequency,
)


@pytest.fixture()
def old_version_qc_report_template(tmp_path):
    with open(DEFAULT_QC_REPORT_TEMPLATE, "r") as f:
        qc_report_template = f.read()

    regex = re.compile(
        r'(?<=<meta name="application-name" content="pixelator-qc-report" data-version=")(.*)(?="/>)'
    )
    qc_report_template_old_version = regex.sub("0.1.0", qc_report_template)

    old_version_template_path = tmp_path / "old_version_qc_report_template.html"
    with open(old_version_template_path, "w") as f:
        f.write(qc_report_template_old_version)

    return old_version_template_path


@pytest.fixture()
def qc_report_data(
    filtered_dataset_pxl_data, raw_component_metrics_data
) -> QCReportData:
    reads_per_molecule_frequency = collect_reads_per_molecule_frequency(
        filtered_dataset_pxl_data
    )
    ranked_component_size_data = collect_component_ranked_component_size_data(
        raw_component_metrics_data
    )

    components_umap_data = collect_components_umap_data(filtered_dataset_pxl_data.adata)
    antibody_percentages_data = collect_antibody_percentages_data(
        filtered_dataset_pxl_data.adata
    )
    antibody_counts_data = collect_antibody_counts_data(filtered_dataset_pxl_data.adata)

    antibodies_per_cell = "antibodies_per_cell test data"
    sequencing_saturation = "sequencing_saturation test data"

    return QCReportData(
        component_data=components_umap_data,
        ranked_component_size=ranked_component_size_data,
        antibodies_per_cell=antibodies_per_cell,
        sequencing_saturation=sequencing_saturation,
        antibody_percentages=antibody_percentages_data,
        antibody_counts=antibody_counts_data,
        reads_per_molecule_frequency=reads_per_molecule_frequency,
    )


@pytest.fixture(scope="module")
def qc_report_metrics(local_assets_dir):
    with open(local_assets_dir / "uropod_control.metrics.json") as f:
        metrics = json.load(f)
        return metrics


def _extract_data(body: LxmlElement, selector: str, decompress: bool = True):
    element = body.cssselect(selector)
    assert len(element) == 1
    data = element[0].text
    if data is None:
        raise RuntimeError(f"No data found for selector {selector}")

    if decompress:
        return gzip.decompress(base64.b64decode(data)).decode("utf-8")

    return data


def test_report_builder_custom_metric_definitions(
    qc_report_metrics, qc_report_data, tmp_path
):
    generation_time = datetime(2023, 1, 1, 0, 0, 0, 0)
    builder = QCReportBuilder()
    sample_info = SampleInfo(
        pixelator_version=__version__,
        generation_date=generation_time.isoformat(),
        sample_id="uropod_control",
        sample_description="just a test",
        pixel_version="1.0.0",
        panel_name="human-sc-immunology-spatial-proteomics",
        panel_version="1.0.0",
        parameters=[],
    )

    metric_definitions_data = {"test": "test"}
    metric_definitions_file = tmp_path / "test-metrics-definitions.json"
    with open(metric_definitions_file, "w") as f:
        json.dump(metric_definitions_data, f)

    with open(Path(tmp_path) / "test.qc-report.html", "wb") as f:
        builder.write(
            f,
            sample_info=sample_info,
            metrics=qc_report_metrics,
            data=qc_report_data,
            metrics_definition_file=metric_definitions_file,
        )

    parser = lxml.etree.HTMLParser(huge_tree=True)
    with open(Path(tmp_path) / "test.qc-report.html", "rb") as f:
        document = lxml.html.parse(f, parser)
        body = document.find("body")

    metrics_data = _extract_data(
        body, 'script[data-type="metric-definitions"]', decompress=True
    )
    assert json.loads(metrics_data) == metric_definitions_data


def test_report_builder_version_mismatch(old_version_qc_report_template):
    with pytest.raises(AssertionError, match="Unsupported QC report version"):
        builder = QCReportBuilder(template=old_version_qc_report_template)
        builder._load_template()


def test_report_builder(qc_report_metrics, qc_report_data, tmp_path):
    """Test the presence of the script tags with the correct data in the HTML report."""
    generation_time = datetime(2023, 1, 1, 0, 0, 0, 0)
    builder = QCReportBuilder()
    sample_info = SampleInfo(
        pixelator_version=__version__,
        generation_date=generation_time.isoformat(),
        sample_id="uropod_control",
        sample_description="just a test",
        pixel_version="1.0.0",
        panel_name="human-sc-immunology-spatial-proteomics",
        panel_version="1.0.0",
        parameters=[],
    )

    res = {
        "info": dataclasses.asdict(sample_info),
        "metrics": qc_report_metrics,
    }

    with open(Path(tmp_path) / "test.qc-report.html", "wb") as f:
        builder.write(
            f,
            sample_info=sample_info,
            metrics=qc_report_metrics,
            data=qc_report_data,
        )

    parser = lxml.etree.HTMLParser(huge_tree=True)
    with open(Path(tmp_path) / "test.qc-report.html", "rb") as f:
        document = lxml.html.parse(f, parser)
        body = document.find("body")

    metrics_data = _extract_data(body, 'script[data-type="metrics"]', decompress=True)
    assert json.loads(metrics_data) == res

    ranked_component_size_data = _extract_data(
        body, 'script[data-type="ranked-component-size"]', decompress=True
    )
    assert ranked_component_size_data == qc_report_data.ranked_component_size

    component_data = _extract_data(
        body, 'script[data-type="component-data"]', decompress=True
    )
    assert component_data == qc_report_data.component_data

    antibody_percentages = _extract_data(
        body, 'script[data-type="antibody-percentages"]', decompress=True
    )
    assert antibody_percentages == qc_report_data.antibody_percentages

    antibody_counts = _extract_data(
        body, 'script[data-type="antibody-counts"]', decompress=True
    )
    assert antibody_counts == qc_report_data.antibody_counts

    reads_per_molecule_frequency = _extract_data(
        body, 'script[data-type="reads-per-molecule-frequency"]', decompress=True
    )
    assert reads_per_molecule_frequency == qc_report_data.reads_per_molecule_frequency

    antibodies_per_cell = _extract_data(
        body, 'script[data-type="antibodies-per-component"]', decompress=False
    )
    assert antibodies_per_cell == qc_report_data.antibodies_per_cell

    sequencing_saturation = _extract_data(
        body, 'script[data-type="sequencing-saturation"]', decompress=False
    )
    assert sequencing_saturation == qc_report_data.sequencing_saturation


@pytest.mark.web_test
@pytest.mark.parametrize("report_file", ["uropod_control_300k_S1_001.qc-report.html"])
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
