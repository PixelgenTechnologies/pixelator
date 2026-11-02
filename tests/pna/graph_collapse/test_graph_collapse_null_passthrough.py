"""Graph-collapse passes a null pxl file through without treating it as a bug.

Copyright © 2026 Pixelgen Technologies AB.
"""

import json

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import PxlFile, write_null_pxl


def test_graph_collapse_cli_passes_null_pxl_through(tmp_path):
    """A null input is copied, the reason is kept, and the step reports failure."""
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
            "graph-collapse",
            str(source.path),
            "--output",
            str(output),
            "--design",
            "proxiome-v1",
            "--report-collapsed-sequences",
        ],
    )
    assert result.exit_code == 0, result.output

    stage_dir = output / "graph-collapse"
    passed = PxlFile(stage_dir / "sampleA.graph-collapse.pxl")
    assert passed.is_null_file()
    assert passed.null_reason() == reason

    report = json.loads((stage_dir / "sampleA.report.json").read_text())
    assert report["status"] == "failed"
    assert report["null_reason"] == reason
    assert report["report_type"] == "graph-collapse"
    assert report["sample_id"] == "sampleA"
    assert report["nodes_collapsed"] == 0
    assert report["collapsed_sequences_file"] is None
    assert not list(stage_dir.glob("*.collapsed_sequences.tsv"))
