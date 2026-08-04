"""Copyright © 2025 Pixelgen Technologies AB."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset import read


def _denoised_output(output_dir, synthetic_denoise_pxl_file) -> Path:
    """Path of the denoised pxl the CLI writes for the synthetic input."""
    return (
        Path(output_dir)
        / "denoise"
        / f"{synthetic_denoise_pxl_file.stem}.denoised_graph.pxl"
    )


def test_denoise_cli_writes_outputs_and_report(synthetic_denoise_pxl_file):
    """The CLI writes the denoised pxl, a report.json and a meta.json.

    The denoising semantics themselves are covered by the in-process tests in
    test_denoise.py; here we only assert what the CLI layer uniquely owns: the
    output file naming, the report metrics, and that the flags are threaded into
    the parameters file.

    Args:
        synthetic_denoise_pxl_file: Synthetic denoise pxl file.
    """
    runner = CliRunner()
    stem = synthetic_denoise_pxl_file.stem

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(synthetic_denoise_pxl_file),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
            "--run-ace-denoising",
            "--run-pls-denoising",
            "--pval-threshold",
            "0.05",
            "--inflate-factor",
            "1.5",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0, cmd.output

        out_pxl = _denoised_output(output_dir, synthetic_denoise_pxl_file)
        assert out_pxl.exists()
        obs = read(out_pxl).adata().obs

        # The report records the UMI removal computed from the written dataset.
        report_path = Path(output_dir) / "denoise" / f"{stem}.report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["report_type"] == "denoise"
        assert report["product_id"] == "single-cell-pna"
        assert report["sample_id"] == stem
        expected_removed = int(obs["number_of_nodes_removed_in_denoise"].sum())
        assert report["number_of_umis_removed"] == expected_removed
        assert report["number_of_umis_removed"] > 0
        assert 0.0 <= report["ratio_of_umis_removed"] <= 1.0

        input_obs = read(synthetic_denoise_pxl_file).adata().obs
        expected_input_reads = int(input_obs["reads_in_component"].sum())
        expected_output_reads = int(obs["reads_in_component"].sum())
        assert report["input_reads"] == expected_input_reads
        assert report["output_reads"] == expected_output_reads
        assert report["output_reads"] <= report["input_reads"]

        # The parameters file records the command and the flags we passed, which
        # confirms the CLI options are threaded into the DenoiseGraph task.
        meta_path = Path(output_dir) / "denoise" / f"{stem}.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["cli"]["command"] == "pixelator single-cell-pna denoise"
        options = meta["cli"]["options"]
        assert options["--run-one-core-graph-denoising"] is True
        assert options["--run-ace-denoising"] is True
        assert options["--run-pls-denoising"] is True
        assert options["--pval-threshold"] == 0.05
        assert options["--inflate-factor"] == 1.5


def test_denoise_cli_no_denoising_method(synthetic_denoise_pxl_file):
    """With no denoising method selected the CLI copies the input and skips work.

    Args:
        synthetic_denoise_pxl_file: Synthetic denoise pxl file.
    """
    runner = CliRunner()
    stem = synthetic_denoise_pxl_file.stem

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(synthetic_denoise_pxl_file),
            "--output",
            output_dir,
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0, cmd.output

        # The report is written with null removal metrics.
        report_path = Path(output_dir) / "denoise" / f"{stem}.report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["report_type"] == "denoise"
        assert report["number_of_umis_removed"] is None
        assert report["ratio_of_umis_removed"] is None

        # No denoising happened, so input and output read counts are identical.
        expected_reads = int(
            read(synthetic_denoise_pxl_file).adata().obs["reads_in_component"].sum()
        )
        assert report["input_reads"] == expected_reads
        assert report["output_reads"] == expected_reads

        # The output pxl is a verbatim copy of the input: denoising was skipped.
        out_pxl = _denoised_output(output_dir, synthetic_denoise_pxl_file)
        assert out_pxl.exists()
        denoised_edges = read(out_pxl).edgelist().to_polars()
        original_edges = read(synthetic_denoise_pxl_file).edgelist().to_polars()
        assert denoised_edges.height == original_edges.height
        assert (set(denoised_edges["umi1"]) | set(denoised_edges["umi2"])) == (
            set(original_edges["umi1"]) | set(original_edges["umi2"])
        )


def test_denoise_cli_missing_input(tmp_path):
    """A non-existent input path is rejected before any work happens.

    Args:
        tmp_path: Tmp path.
    """
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "single-cell-pna",
            "denoise",
            str(tmp_path / "does_not_exist.pxl"),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code != 0


def test_denoise_cli_rejects_non_pxl_input(tmp_path):
    """An input file without a .pxl extension is rejected by the sanity check.

    Args:
        tmp_path: Tmp path.
    """
    runner = CliRunner()
    bad_input = tmp_path / "not_a_pixelfile.txt"
    bad_input.write_text("not a pxl file")

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "single-cell-pna",
            "denoise",
            str(bad_input),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code != 0
