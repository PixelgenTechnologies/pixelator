"""Analysis passes a null pxl file through without treating it as a bug."""

import json

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import PxlFile, write_null_pxl


def test_analysis_cli_passes_null_pxl_through(tmp_path):
    """A null input is copied, the reason is kept, and the parameters file is written."""
    reason = "No cells above the component size threshold."
    source = write_null_pxl(
        tmp_path / "sampleA.graph.pxl",
        sample_name="sampleA",
        reason=reason,
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "analysis",
            str(source.path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output

    passed = PxlFile(output / "analysis" / "sampleA.analysis.pxl")
    assert passed.is_null_file()
    assert passed.null_reason() == reason

    report = json.loads((output / "analysis" / "sampleA.report.json").read_text())
    assert report["status"] == "failed"
    assert report["null_reason"] == reason
    assert report["sample_id"] == "sampleA"
    assert (output / "analysis" / "sampleA.meta.json").is_file()
