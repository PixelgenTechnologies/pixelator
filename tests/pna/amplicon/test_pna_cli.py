"""Copyright © 2025 Pixelgen Technologies AB."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zstandard as zstd
from click.testing import CliRunner

from pixelator import cli


@pytest.fixture()
def testdata_300k_sample_mismatch(testdata_300k):
    [r1, r2] = testdata_300k

    pwd = Path.cwd()

    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        r1_cpy = shutil.copy(r1, Path(d) / "testdata_300k_R1.fq.zst")
        r2_cpy = shutil.copy(r2, Path(d) / "testdata_300k_typo_R2.fq.zst")

        yield (r1_cpy, r2_cpy)

    os.chdir(pwd)


@pytest.mark.slow
def test_fastq_valid_inputs(testdata_300k):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "amplicon",
            str(testdata_300k[0]),
            str(testdata_300k[1]),
            "--output",
            str(d),
            "--design",
            "pna-2",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0


def test_fastq_swapped_read_input(testdata_300k):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "amplicon",
            str(testdata_300k[1]),
            str(testdata_300k[0]),
            "--output",
            str(d),
            "--design",
            "pna-2",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code != 0
        assert (
            "ERROR: Read 1 file does not contain a recognised read 1 suffix."
            in cmd.output
        )


def test_fastq_sample_name_mismatch(testdata_300k_sample_mismatch):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "amplicon",
            str(testdata_300k_sample_mismatch[0]),
            str(testdata_300k_sample_mismatch[1]),
            "--output",
            str(d),
            "--design",
            "pna-2",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code != 0
        assert "ERROR: The sample name for read1 and read2 is different" in cmd.output


def test_can_skip_input_checks(mocker, testdata_300k_sample_mismatch):
    runner = CliRunner()

    def mock_amplicon_sample_report():
        def f(**kwargs):
            m = MagicMock()
            m.fraction_discarded_reads = 0.1
            return m

        return f

    # Patch amplicon_fastq to be a no-op to speed this up
    mocker.patch("pixelator.pna.cli.amplicon.amplicon_fastq")
    mocker.patch(
        "pixelator.pna.cli.amplicon.AmpliconSampleReport",
        new_callable=mock_amplicon_sample_report,
    )

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "amplicon",
            str(testdata_300k_sample_mismatch[0]),
            str(testdata_300k_sample_mismatch[1]),
            "--output",
            str(d),
            "--design",
            "pna-2",
            "--skip-input-checks",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0


@pytest.mark.slow
def test_fastq_single_end(testdata_unbalanced_r12):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "amplicon",
            str(testdata_unbalanced_r12[1]),
            "--output",
            str(d),
            "--design",
            "pna-2",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        def read_zst_lines(path):
            with open(path, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    return reader.read().splitlines()

        reads = read_zst_lines(d + "/amplicon/unbalanced.amplicon.fq.zst")
        assert len(reads) % 4 == 0
        assert len(reads) > 300
