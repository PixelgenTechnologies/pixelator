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

REFERENCE_COMPONENT = "57129a8b0fff38c6"
ACE_LOW_NODE_COUNT = 5436


@pytest.mark.slow
def test_denoise_runs_ok(denoise_pxl_file):
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(denoise_pxl_file),
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

        result = read(Path(output_dir) / "denoise" / "test_denoise.denoised_graph.pxl")
        assert not result.adata().obs["disqualified_for_denoising"].any()


@pytest.mark.slow
def test_denoise_ace_cli_runs_ok(denoise_pxl_file):
    """ACE-only denoise completes and records ACE-specific removal counts."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(denoise_pxl_file),
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

        out_pxl = Path(output_dir) / "denoise" / "test_denoise.denoised_graph.pxl"
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
def test_denoise_one_core_and_ace_cli_runs_ok(denoise_pxl_file):
    """One-core plus ACE: ACE counts match full-graph ACE; total includes one-core and stranding."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(denoise_pxl_file),
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

        out_pxl = Path(output_dir) / "denoise" / "test_denoise.denoised_graph.pxl"
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


@pytest.mark.slow
def test_denoise_ace_pls_cli_runs_ok(denoise_pxl_file):
    """One-core plus ACE: ACE counts match full-graph ACE; total includes one-core and stranding."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as output_dir:
        args = [
            "--cores",
            "1",
            "single-cell-pna",
            "denoise",
            str(denoise_pxl_file),
            "--output",
            output_dir,
            "--run-ace-denoising",
            "--run-pls-denoising",
        ]
        cmd = runner.invoke(cli.main_cli, args)
        assert cmd.exit_code == 0, cmd.output

        out_pxl = Path(output_dir) / "denoise" / "test_denoise.denoised_graph.pxl"
        result = read(out_pxl)
        markers = result.adata().var.index
        isotype_controls = markers[markers.str.contains("mIgG")]
        original_isotype_levels = (
            read(pna_pxl_file).adata().obsm["clr"].loc[:, isotype_controls]
        ).mean(axis=1)
        result_isotype_levels = (
            result.adata().obsm["clr"].loc[:, isotype_controls].mean(axis=1)
        )
        assert (
            result_isotype_levels
            < original_isotype_levels.reindex(result_isotype_levels.index)
        ).all()
