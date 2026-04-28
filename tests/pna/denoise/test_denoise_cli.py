"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.pna.config import pna_config
from pixelator.pna.config.panel import load_antibody_panel
from pixelator.pna.pixeldataset import read

REFERENCE_COMPONENT = "0a45497c6bfbfb22"
ACE_LOW_NODE_COUNT = 399


@pytest.mark.slow
def test_denoise_runs_ok(pxl_file):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(pxl_file),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
            "--pval-threshold",
            "0.05",
            "--inflate-factor",
            "1.5",
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        result = read(Path(output_dir) / "denoise" / "file.denoised_graph.pxl")
        assert not result.adata().obs["disqualified_for_denoising"].any()


def _patch_denoise_panel_loader():
    """Layout test PXLs may lack ``panel_metadata``; mirror ``test_denoise_one_core_analysis``."""

    def _load(*args, **kwargs):
        return load_antibody_panel(pna_config, "proxiome-immuno-155-v2")

    return mock.patch(
        "pixelator.pna.analysis.denoise.load_antibody_panel", side_effect=_load
    )


@pytest.mark.slow
def test_denoise_ace_cli_runs_ok(pna_pxl_file):
    """ACE-only denoise completes and records ACE-specific removal counts."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as output_dir, _patch_denoise_panel_loader():
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(pna_pxl_file),
            "--output",
            output_dir,
            "--run-ace-denoising",
            "--ace-k",
            "3",
            "--ace-max-k-core",
            "4",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0, cmd.output

        out_pxl = Path(output_dir) / "denoise" / "PNA055_Sample07_S7.denoised_graph.pxl"
        result = read(out_pxl)
        obs = result.adata().obs
        assert "number_of_nodes_removed_in_denoise_ace" in obs.columns
        assert (
            int(obs.loc[REFERENCE_COMPONENT, "number_of_nodes_removed_in_denoise_ace"])
            == ACE_LOW_NODE_COUNT
        )
        assert (
            int(obs.loc[REFERENCE_COMPONENT, "number_of_nodes_removed_in_denoise"])
            == ACE_LOW_NODE_COUNT
        )


@pytest.mark.slow
def test_denoise_one_core_and_ace_cli_runs_ok(pna_pxl_file):
    """One-core then ACE (CLI order) produces cumulative removal stats."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as output_dir, _patch_denoise_panel_loader():
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(pna_pxl_file),
            "--output",
            output_dir,
            "--run-one-core-graph-denoising",
            "--run-ace-denoising",
            "--pval-threshold",
            "0.05",
            "--inflate-factor",
            "1.5",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0, cmd.output

        out_pxl = Path(output_dir) / "denoise" / "PNA055_Sample07_S7.denoised_graph.pxl"
        result = read(out_pxl)
        obs = result.adata().obs
        assert "number_of_nodes_removed_in_denoise_ace" in obs.columns
        ace_removed = int(
            obs.loc[REFERENCE_COMPONENT, "number_of_nodes_removed_in_denoise_ace"]
        )
        total_removed = int(
            obs.loc[REFERENCE_COMPONENT, "number_of_nodes_removed_in_denoise"]
        )
        assert ace_removed == ACE_LOW_NODE_COUNT
        assert total_removed >= ace_removed
        assert total_removed > 0
