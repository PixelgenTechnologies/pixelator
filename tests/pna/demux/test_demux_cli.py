"""Copyright © 2025 Pixelgen Technologies AB."""

from click.testing import CliRunner

from pixelator import cli


def test_demux_invalid_chunk_size(tmp_path, testdata_amplicon_fastq):
    runner = CliRunner()

    args = [
        "single-cell-pna",
        "demux",
        str(testdata_amplicon_fastq),
        "--output",
        str(tmp_path),
        "--design",
        "pna-2",
        "--panel",
        "proxiome-immuno-155",
        "--output-chunk-reads",
        "123239KB",
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 2
    assert "Invalid value for '--output-chunk-reads'" in cmd.output


def test_demux_valid(tmp_path, testdata_amplicon_fastq):
    runner = CliRunner()

    args = [
        "single-cell-pna",
        "demux",
        str(testdata_amplicon_fastq),
        "--output",
        str(tmp_path),
        "--design",
        "pna-2",
        "--panel",
        "proxiome-immuno-155",
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 0
