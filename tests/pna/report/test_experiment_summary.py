"""Experiment summary includes failed samples and earlier statistics."""

import json

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset.io import write_null_pxl
from pixelator.pna.report.experiment_summary import build_experiment_summary


def _write_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_experiment_summary_reports_failed_step_and_keeps_earlier_stats(tmp_path):
    """A sample that fails at graph stays in the summary with upstream stats."""
    reason = "No connected components found in the graph."
    _write_report(
        tmp_path / "amplicon" / "sampleA.report.json",
        {
            "sample_id": "sampleA",
            "product_id": "single-cell-pna",
            "report_type": "amplicon",
            "input_reads": 1000,
            "output_reads": 900,
        },
    )
    _write_report(
        tmp_path / "graph" / "sampleA.report.json",
        {
            "sample_id": "sampleA",
            "product_id": "single-cell-pna",
            "report_type": "graph",
            "status": "failed",
            "null_reason": reason,
            "reads_input": 900,
            "molecules_input": 80,
        },
    )
    write_null_pxl(
        tmp_path / "graph" / "sampleA.graph.pxl",
        sample_name="sampleA",
        reason=reason,
    )
    _write_report(
        tmp_path / "denoise" / "sampleA.report.json",
        {
            "sample_id": "sampleA",
            "product_id": "single-cell-pna",
            "report_type": "denoise",
            "status": "failed",
            "null_reason": reason,
            "input_reads": 0,
            "output_reads": 0,
        },
    )
    write_null_pxl(
        tmp_path / "denoise" / "sampleA.denoised_graph.pxl",
        sample_name="sampleA",
        reason=reason,
    )
    _write_report(
        tmp_path / "graph" / "sampleB.report.json",
        {
            "sample_id": "sampleB",
            "product_id": "single-cell-pna",
            "report_type": "graph",
            "reads_input": 50,
        },
    )

    summary = build_experiment_summary(tmp_path)
    by_id = {sample.sample_id: sample for sample in summary.samples}

    failed = by_id["sampleA"]
    assert failed.status == "failed"
    assert failed.null_reason == reason
    steps = {step.step: step for step in failed.steps}
    assert steps["amplicon"].status == "passed"
    assert steps["amplicon"].statistics["input_reads"] == 1000
    assert steps["amplicon"].statistics["output_reads"] == 900
    assert steps["graph"].status == "failed"
    assert steps["graph"].null_reason == reason
    assert steps["graph"].statistics["reads_input"] == 900
    assert steps["graph"].statistics["molecules_input"] == 80
    assert steps["denoise"].status == "failed"
    assert steps["denoise"].null_reason == reason
    assert steps["layout"].status == "not_run"
    assert "graph: failed" in failed.describe()
    assert "amplicon: passed" in failed.describe()

    assert by_id["sampleB"].status == "passed"
    assert by_id["sampleB"].null_reason is None


def test_experiment_summary_cli_writes_json(tmp_path):
    """The CLI writes the summary next to the other stage outputs."""
    reason = "No cells above the component size threshold."
    _write_report(
        tmp_path / "input" / "graph" / "sampleA.report.json",
        {
            "sample_id": "sampleA",
            "report_type": "graph",
            "status": "failed",
            "null_reason": reason,
            "reads_input": 12,
        },
    )
    write_null_pxl(
        tmp_path / "input" / "graph" / "sampleA.graph.pxl",
        sample_name="sampleA",
        reason=reason,
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "single-cell-pna",
            "experiment-summary",
            str(tmp_path / "input"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    written = json.loads(
        (output / "experiment_summary" / "experiment_summary.json").read_text()
    )
    sample = written["samples"][0]
    assert sample["sample_id"] == "sampleA"
    assert sample["status"] == "failed"
    assert sample["null_reason"] == reason
    graph_step = next(step for step in sample["steps"] if step["step"] == "graph")
    assert graph_step["status"] == "failed"
    assert graph_step["statistics"]["reads_input"] == 12
