"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile

import pytest
from click.testing import CliRunner

from pixelator import cli


@pytest.mark.skip(reason="For debugging only")
def test_collapse_run(mocker, testdata_paired_small_demux):
    """Verify collapse run.

    Args:
    mocker: mocker.
    testdata_paired_small_demux: testdata paired small demux.

    """
    runner = CliRunner()
    mocker.patch("pixelator.cli.amplicon.amplicon_fastq")

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "collapse",
            str(testdata_paired_small_demux),
            "--output",
            str(d),
            "--design",
            "proxiome-v1",
            "--panel",
            "proxiome-v1-immuno-155-v1.1",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0


@pytest.mark.parametrize("panel_file", ["proxiome-v1-immuno-155-v1.0"])
def test_invalid_mismatch_param(mocker, testdata_paired_small_demux, panel_file):
    """Verify invalid mismatch param.

    Args:
    mocker: mocker.
    testdata_paired_small_demux: testdata paired small demux.
    panel_file: panel file.

    """
    runner = CliRunner()
    mocker.patch("pixelator.pna.cli.amplicon.amplicon_fastq")

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "collapse",
            str(testdata_paired_small_demux),
            "--output",
            str(d),
            "--design",
            "proxiome-v1",
            "--panel",
            panel_file,
            "--mismatches",
            "13.5",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code != 0
        assert "Must be an integer >= 1 or a float in range [0, 1)" in cmd.output
