"""Tests for the sample calling CLI.

Copyright © 2025 Pixelgen Technologies AB.
"""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna import read


@pytest.mark.slow
def test_runs_ok(mocker, pna_data_root):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "single-cell-pna",
            "sample-calling",
            str(pna_data_root / "sample_calling/small_hashed_sample_1.pxl"),
            "--samplesheet",
            str(pna_data_root) + "/sample_calling/samplesheet.csv",
            "--panel",
            "sample-calling-test-panel",
            "--remove-incompatible",
            "--confidence-threshold",
            "0.96",
            "--save-undetermined",
            "--output",
            d,
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        outputs = list((Path(d) / "sample_calling").glob("*.dehashed.pxl"))
        total_components = 0
        for output in outputs:
            pxl = read(output)
            total_components += pxl.adata().obs.shape[0]
        assert total_components == 15
