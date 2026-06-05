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
        "proxiome-v1",
        "--panel",
        "proxiome-v1-immuno-155-v1.1",
        "--output-chunk-reads",
        "123239KB",
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 2
    assert "Invalid value for '--output-chunk-reads'" in cmd.output


@pytest.mark.slow
@pytest.mark.parametrize("panel_file", ["proxiome-v1-immuno-155-v1.0"])
def test_demux_valid(tmp_path, testdata_amplicon_fastq, panel_file):
    runner = CliRunner()

    args = [
        "single-cell-pna",
        "demux",
        str(testdata_amplicon_fastq),
        "--output",
        str(tmp_path),
        "--design",
        "proxiome-v1",
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
        "proxiome-v1",
        "--panel",
        str(pna_data_root / "panels/test-pna-panel.csv"),
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 0


@pytest.mark.slow
def test_demux_two_panels(tmp_path, testdata_amplicon_fastq, pna_data_root):
    """Verify demux accepts repeated --panel for base and hashing panels."""
    runner = CliRunner()
    hashing_csv = tmp_path / "hash-panel.csv"
    hashing_csv.write_text(
        """# ---
# name: hash-set-1
# product: hash-set-1
# version: 0.1.0
# panel_type: sample_hashing
# ---
marker_id,control,sequence_1,sequence_2,sample_hashing
HM-1,no,ACTTCCTACC,ACTTCCTACC,yes
HM-2,no,GGGCTATGGT,GGGCTATGGT,yes
"""
    )

    args = [
        "single-cell-pna",
        "demux",
        str(testdata_amplicon_fastq),
        "--output",
        str(tmp_path / "demux_out"),
        "--design",
        "proxiome-v1",
        "--panel",
        str(pna_data_root / "panels/test-pna-panel.csv"),
        "--panel",
        str(hashing_csv),
    ]
    cmd = runner.invoke(cli.main_cli, args)

    assert cmd.exit_code == 0
