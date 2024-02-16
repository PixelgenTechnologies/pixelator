"""
Test the pixelator single-cell amplicon CLI.

Copyright © 2022 Pixelgen Technologies AB.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from pixelator import cli


@pytest.fixture()
def uropod_reads(data_root):
    """Paired end reads from Uropod control sample."""
    r1 = data_root / "uropod_control_300k_S1_R1_001.fastq.gz"
    r2 = data_root / "uropod_control_300k_S1_R2_001.fastq.gz"
    return r1, r2


@pytest.fixture()
def run_in_tmpdir():
    """Run a function in a temporary directory."""
    old_cwd = Path.cwd()

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        yield d
        os.chdir(old_cwd)


@pytest.fixture()
def uropod_reads_sample_mismatch(data_root):
    r1 = data_root / "uropod_control_300k_S1_R1_001.fastq.gz"
    r2 = data_root / "uropod_control_300k_S1_R2_001.fastq.gz"

    pwd = Path.cwd()

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        r1_cpy = shutil.copy(r1, Path(d) / "uropod_control_R1.fastq.gz")
        r2_cpy = shutil.copy(r2, Path(d) / "uropod_typo_R2.fastq.gz")

        yield (r1_cpy, r2_cpy)

    os.chdir(pwd)


def test_fastq_valid_inputs(mocker, uropod_reads, run_in_tmpdir):
    runner = CliRunner()

    mocker.patch("pixelator.cli.amplicon.amplicon_fastq")

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell",
            "amplicon",
            str(uropod_reads[0]),
            str(uropod_reads[1]),
            "--output",
            str(d),
            "--design",
            "D21",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0

        cmd = runner.invoke(cli.main_cli, args + ["--skip-input-checks"])
        assert cmd.exit_code == 0


def test_fastq_swapped_read_input(mocker, uropod_reads, tmp_path):
    runner = CliRunner()

    mocker.patch("pixelator.cli.amplicon.amplicon_fastq")

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell",
            "amplicon",
            str(uropod_reads[1]),
            str(uropod_reads[0]),
            "--output",
            str(d),
            "--design",
            "D21",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code != 0
        assert (
            "ERROR: Read 1 file does not contain a recognised read 1 suffix."
            in cmd.output
        )

        cmd = runner.invoke(
            cli.main_cli,
            [
                "single-cell",
                "amplicon",
                str(uropod_reads[0]),
                str(uropod_reads[0]),
                "--output",
                str(d),
                "--design",
                "D21",
            ],
        )

        assert (
            "ERROR: Read 2 file does not contain a recognised read 2 suffix."
            in cmd.output
        )

        cmd = runner.invoke(cli.main_cli, args + ["--skip-input-checks"])
        assert cmd.exit_code == 0


def test_fastq_sample_name_mismatch(mocker, uropod_reads_sample_mismatch):
    runner = CliRunner()
    mocker.patch("pixelator.cli.amplicon.amplicon_fastq")

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell",
            "amplicon",
            str(uropod_reads_sample_mismatch[0]),
            str(uropod_reads_sample_mismatch[1]),
            "--output",
            str(d),
            "--design",
            "D21",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code != 0
        assert "ERROR: The sample name for read1 and read2 is different" in cmd.output

        cmd = runner.invoke(cli.main_cli, args + ["--skip-input-checks"])
        assert cmd.exit_code == 0
