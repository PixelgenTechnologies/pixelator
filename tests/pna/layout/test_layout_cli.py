"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path

import networkx as nx
import polars as pl
import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.layout import CreateLayout
from pixelator.pna.pixeldataset import read


@pytest.mark.slow
def test_runs_ok(pna_data_root):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        args = [
            "--cores",
            "1",  # Notabene, using 1 thread here to make sure we actually use the mock
            "single-cell-pna",
            "layout",
            str(pna_data_root / "PNA055_Sample07_S7.layout.pxl"),
            "--output",
            d,
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        result = read(Path(d) / "layout" / "PNA055_Sample07_S7.layout.pxl")
        assert not result.precomputed_layouts().is_empty()
