"""Copyright © 2025 Pixelgen Technologies AB."""

import pytest
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "panel_file", ["proxiome-immuno-155-v1", "proxiome-immuno-155-v2"]
)
def test_demux_valid(tmp_path, testdata_amplicon_fastq, panel_file):
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
        panel_file,
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 0


@pytest.mark.slow
def test_demux_custom_panel(tmp_path, testdata_amplicon_fastq, pna_data_root):
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
        str(pna_data_root / "panels/test-pna-panel.csv"),
    ]
    cmd = runner.invoke(cli.main_cli, args)

    print(cmd.output)

    assert cmd.exit_code == 0
