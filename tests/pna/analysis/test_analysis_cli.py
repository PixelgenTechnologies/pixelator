"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset import read


@pytest.mark.slow
def test_runs_ok(mocker, pna_data_root):
    runner = CliRunner()

    # Picking a two components here to run to speed up the computations...
    def downsample_read():
        def f(paths):
            pxl_dataset = read(paths)
            single_component = list(pxl_dataset.components())[:2]
            return pxl_dataset.filter(components=single_component)

        return f

    mocker.patch("pixelator.pna.cli.analysis.read", new_callable=downsample_read)

    with tempfile.TemporaryDirectory() as d:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "analysis",
            str(pna_data_root / "PNA055_Sample07_S7.layout.pxl"),
            "--output",
            d,
            "--compute-proximity",
            "--proximity-nbr-of-permutations",
            "50",
            "--compute-svd-var-explained",
            "--svd-nbr-of-pivots",
            "10",
            "--compute-k-cores",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        result = read(Path(d) / "analysis" / "PNA055_Sample07_S7.analysis.pxl")
        assert not result.proximity().is_empty()

        adata = result.adata()

        assert "k_core_1" in adata.obs.columns
        assert "k_core_2" in adata.obs.columns
        assert "k_core_3" in adata.obs.columns
        assert "average_k_core" in adata.obs.columns

        assert "svd_var_expl_s1" in adata.obs.columns
        assert "svd_var_expl_s2" in adata.obs.columns
        assert "svd_var_expl_s3" in adata.obs.columns
