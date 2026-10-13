"""Copyright © 2025 Pixelgen Technologies AB."""

import shutil

from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.pixeldataset import read


def test_runs_ok(synthetic_analysis_pxl_file, tmp_path):
    """The analysis command adds proximity, SVD and k-core results to the pxl file.

    Args:
        synthetic_analysis_pxl_file: synthetic analysis pxl file.
        tmp_path: tmp path.
    """
    # The output file name is derived from the input file name, so the input is
    # copied under the name of a real layout file.
    input_pxl = tmp_path / "PNA055_Sample07_S7.layout.pxl"
    shutil.copy(synthetic_analysis_pxl_file, input_pxl)
    output_dir = tmp_path / "output"

    args = [
        "--cores",
        "1",
        "single-cell-pna",
        "analysis",
        str(input_pxl),
        "--output",
        str(output_dir),
        "--compute-proximity",
        "--proximity-nbr-of-permutations",
        "50",
        "--compute-svd-var-explained",
        "--svd-nbr-of-pivots",
        "10",
        "--compute-k-cores",
    ]
    result = CliRunner().invoke(cli.main_cli, args)

    assert result.exit_code == 0

    analyzed = read(output_dir / "analysis" / "PNA055_Sample07_S7.analysis.pxl")

    # `--compute-proximity`
    assert not analyzed.proximity().is_empty()

    # `--compute-k-cores`
    obs = analyzed.adata().obs
    assert "k_core_1" in obs.columns
    assert "k_core_2" in obs.columns
    assert "k_core_3" in obs.columns
    assert "average_k_core" in obs.columns

    # `--compute-svd-var-explained`
    assert "svd_var_expl_s1" in obs.columns
    assert "svd_var_expl_s2" in obs.columns
    assert "svd_var_expl_s3" in obs.columns
