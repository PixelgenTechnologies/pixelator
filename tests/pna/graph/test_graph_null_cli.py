"""Graph CLI behavior for recoverable and unrecoverable failures."""

import json

import polars as pl
import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.graph.component_recovery_utils import ConnectedComponentException
from pixelator.pna.graph.report import GraphStatistics
from pixelator.pna.pixeldataset.io import PxlFile


def _parquet(tmp_path):
    path = tmp_path / "sampleA.parquet"
    pl.DataFrame({"umi1": [1]}).write_parquet(path)
    return path


def test_graph_cli_writes_null_pxl_when_no_cells_pass(tmp_path, monkeypatch):
    """A data-caused component failure emits a null file and a failed report."""
    reason = "No connected components found in the graph."

    def raise_recoverable(*args, **kwargs):
        raise ConnectedComponentException(
            reason,
            statistics=GraphStatistics(reads_input=42, molecules_input=7),
        )

    monkeypatch.setattr(
        "pixelator.pna.cli.graph.build_pxl_file_with_components",
        raise_recoverable,
    )
    runner = CliRunner()
    output = tmp_path / "out"
    result = runner.invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph",
            str(_parquet(tmp_path)),
            "--panel",
            "proxiome-v1-immuno-155-v1.1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output

    pxl_path = output / "graph" / "sampleA.graph.pxl"
    pxl_file = PxlFile(pxl_path)
    assert pxl_file.is_null()
    assert pxl_file.null_reason() == reason

    report = json.loads((output / "graph" / "sampleA.report.json").read_text())
    assert report["status"] == "failed"
    assert report["null_reason"] == reason
    assert report["reads_input"] == 42
    assert report["molecules_input"] == 7
    assert report["sample_id"] == "sampleA"


def test_graph_cli_still_fails_on_unexpected_exception(tmp_path, monkeypatch):
    """Software bugs are not converted into null pxl files."""

    def raise_bug(*args, **kwargs):
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(
        "pixelator.pna.cli.graph.build_pxl_file_with_components",
        raise_bug,
    )
    runner = CliRunner()
    result = runner.invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph",
            str(_parquet(tmp_path)),
            "--panel",
            "proxiome-v1-immuno-155-v1.1",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "out" / "graph" / "sampleA.graph.pxl").exists()


def test_connected_component_exception_keeps_statistics():
    """The recoverable exception carries the stats gathered before it was raised."""
    stats = GraphStatistics(reads_input=3)
    exc = ConnectedComponentException("no cells", statistics=stats)
    assert exc.statistics is stats
    with pytest.raises(ConnectedComponentException):
        raise exc
