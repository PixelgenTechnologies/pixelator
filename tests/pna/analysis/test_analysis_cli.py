"""Copyright © 2025 Pixelgen Technologies AB."""

import shutil
import tempfile
from pathlib import Path

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset import read


def test_runs_ok(synthetic_analysis_pxl_file, tmp_path):
    """Verify runs ok.

    Args:
        synthetic_analysis_pxl_file: synthetic analysis pxl file.
        tmp_path: tmp path.
    """
    runner = CliRunner()

    # The output file name is derived from the input file name.
    input_pxl = tmp_path / "PNA055_Sample07_S7.layout.pxl"
    shutil.copy(synthetic_analysis_pxl_file, input_pxl)

    with tempfile.TemporaryDirectory() as d:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "analysis",
            str(input_pxl),
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
