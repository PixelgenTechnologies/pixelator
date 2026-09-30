"""Sample calling keeps samplesheet samples when the input pxl is null."""

import json

import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import PxlFile, write_null_pxl


def test_sample_calling_emits_null_file_per_samplesheet_sample(tmp_path):
    """Each samplesheet sample keeps the pool reason and panel metadata."""
    reason = "No cells above the component size threshold."
    source = write_null_pxl(
        tmp_path / "poolA.graph.pxl",
        sample_name="poolA",
        reason=reason,
        panel_name="proxiome-v1",
        panel_version="1.2.3",
        source_metadata={"panel_alias": "immuno"},
    )
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        "pool,sample,hash_index\npoolA,sample1,1\npoolA,sample2,2\nother,sample3,1\n"
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "sample-calling",
            str(source.path),
            "--samplesheet",
            str(samplesheet),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output

    for sample_name in ("sample1", "sample2"):
        pxl_file = PxlFile(output / "sample_calling" / f"{sample_name}.dehashed.pxl")
        assert pxl_file.is_null_file()
        assert pxl_file.null_reason() == reason
        assert pxl_file.sample_name == sample_name
        metadata = pxl_file.metadata()
        assert metadata["panel_name"] == "proxiome-v1"
        assert metadata["panel_version"] == "1.2.3"
        assert metadata["panel_alias"] == "immuno"
        assert metadata["sample_name"] == sample_name
        report = json.loads(
            (output / "sample_calling" / f"{sample_name}.report.json").read_text()
        )
        assert report["status"] == "failed"
        assert report["null_reason"] == reason

    assert not (output / "sample_calling" / "sample3.dehashed.pxl").exists()


@pytest.mark.parametrize(
    "reserved_name",
    ["undetermined", "poolA_undetermined"],
)
def test_null_sample_calling_rejects_reserved_sample_names(tmp_path, reserved_name):
    """A null input still refuses samplesheet names reserved for undetermined components."""
    source = write_null_pxl(
        tmp_path / "poolA.graph.pxl",
        sample_name="poolA",
        reason="No cells above the component size threshold.",
    )
    samplesheet = tmp_path / "samplesheet.csv"
    samplesheet.write_text(
        f"pool,sample,hash_index\npoolA,{reserved_name},1\npoolA,sample1,2\n"
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "sample-calling",
            str(source.path),
            "--samplesheet",
            str(samplesheet),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "not allowed in the samplesheet" in result.output
    assert not (output / "sample_calling" / f"{reserved_name}.dehashed.pxl").exists()
    assert not (output / "sample_calling" / "sample1.dehashed.pxl").exists()
