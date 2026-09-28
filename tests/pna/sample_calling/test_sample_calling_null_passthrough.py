"""Sample calling keeps samplesheet samples when the input pxl is null."""

import json

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import PxlFile, write_null_pxl


def test_sample_calling_emits_null_file_per_samplesheet_sample(tmp_path):
    """Each samplesheet sample in the pool gets a null file with the same reason."""
    reason = "No cells above the component size threshold."
    source = write_null_pxl(
        tmp_path / "poolA.graph.pxl",
        sample_name="poolA",
        reason=reason,
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
        assert pxl_file.is_null()
        assert pxl_file.null_reason() == reason
        assert pxl_file.sample_name == sample_name
        report = json.loads(
            (output / "sample_calling" / f"{sample_name}.report.json").read_text()
        )
        assert report["status"] == "failed"
        assert report["null_reason"] == reason

    assert not (output / "sample_calling" / "sample3.dehashed.pxl").exists()
