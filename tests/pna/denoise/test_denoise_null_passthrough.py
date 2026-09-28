"""Denoise passes a null pxl file through without treating it as a bug."""

import json

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import PxlFile, write_null_pxl


def test_denoise_cli_passes_null_pxl_through(tmp_path):
    """A null input is copied, the reason is kept, and the step reports failure."""
    reason = "No cells above the component size threshold."
    source = write_null_pxl(
        tmp_path / "sampleA.graph.pxl",
        sample_name="sampleA",
        reason=reason,
    )
    output = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(source.path),
            "--output",
            str(output),
            "--run-one-core-graph-denoising",
        ],
    )
    assert result.exit_code == 0, result.output

    passed = PxlFile(output / "denoise" / "sampleA.denoised_graph.pxl")
    assert passed.is_null()
    assert passed.null_reason() == reason

    report = json.loads((output / "denoise" / "sampleA.report.json").read_text())
    assert report["status"] == "failed"
    assert report["null_reason"] == reason
    assert report["report_type"] == "denoise"
    assert report["sample_id"] == "sampleA"
