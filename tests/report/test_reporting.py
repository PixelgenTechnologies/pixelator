"""Copyright © 2023 Pixelgen Technologies AB."""
import pytest

from pixelator.report import PixelatorReporting, PixelatorWorkdir


def test_reporting_plain_dir_constructor(pixelator_workdir):
    # Construct from Path instance
    reporting = PixelatorReporting(pixelator_workdir.basedir)
    assert isinstance(reporting.workdir, PixelatorWorkdir)


def test_reporting_samples(pixelator_workdir, all_stages_all_reports_and_meta):
    # Construct from PixelatorWorkdir instance
    reporting = PixelatorReporting(pixelator_workdir)
    assert reporting.samples() == {
        "pbmcs_unstimulated",
        "uropod_control",
    }


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_reporting_reads_flow(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot, sample_name
):
    reporting = PixelatorReporting(pixelator_workdir)
    reads_flow = reporting.reads_flow(sample_name)

    snapshot.assert_match(
        reads_flow.to_json(indent=4), f"{sample_name}_reads_flow.json"
    )


@pytest.mark.parametrize("sample_name", ["pbmcs_unstimulated", "uropod_control"])
def test_reporting_molecules_flow(
    pixelator_workdir, all_stages_all_reports_and_meta, snapshot, sample_name
):
    reporting = PixelatorReporting(pixelator_workdir)
    molecules_flow = reporting.molecules_flow(sample_name)

    snapshot.assert_match(
        molecules_flow.to_json(indent=4), f"{sample_name}_molecules_flow.json"
    )
