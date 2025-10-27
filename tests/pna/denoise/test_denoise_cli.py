"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path
import tempfile

from click.testing import CliRunner
import pytest

from pixelator import cli
from pixelator.pna.pixeldataset import read


def test_denoise_runs_ok(pxl_file):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",  # Notabene, using 1 thread here to make sure we actually use the mock
            "single-cell-pna",
            "denoise",
            str(pxl_file),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
            "--pval-threshold",
            "0.05",
            "--inflate-factor",
            "1.5",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        result = read(Path(output_dir) / "denoise" / "file.denoised_graph.pxl")
        assert not result.adata().obs["disqualified_for_denoising"].any()
