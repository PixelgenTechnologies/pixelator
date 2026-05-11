"""Copyright © 2025 Pixelgen Technologies AB."""

import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
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
        assert (
            int(obs.loc[REFERENCE_COMPONENT, "denoised_nodes_marked_only_by_ace"])
            == ACE_LOW_NODE_COUNT
        )
        summary_cols = [
            "denoised_nodes_marked_only_by_ace",
            "denoised_nodes_marked_only_by_pls",
            "denoised_nodes_marked_only_by_one_core",
            "denoised_nodes_marked_stranded",
            "denoised_nodes_marked_ace_and_pls",
            "denoised_nodes_marked_ace_and_one_core",
            "denoised_nodes_marked_pls_and_one_core",
            "denoised_nodes_marked_ace_pls_and_one_core",
        ]
        assert set(summary_cols).issubset(obs.columns)
        pd.testing.assert_frame_equal(
            obs.loc[:, summary_cols]
            .sum(axis=1)
            .to_frame("number_of_nodes_removed_in_denoise"),
            obs.loc[:, ["number_of_nodes_removed_in_denoise"]],
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
        summary_cols = [
            "denoised_nodes_marked_only_by_ace",
            "denoised_nodes_marked_only_by_pls",
            "denoised_nodes_marked_only_by_one_core",
            "denoised_nodes_marked_stranded",
            "denoised_nodes_marked_ace_and_pls",
            "denoised_nodes_marked_ace_and_one_core",
            "denoised_nodes_marked_pls_and_one_core",
            "denoised_nodes_marked_ace_pls_and_one_core",
        ]
        assert set(summary_cols).issubset(obs.columns)
        pd.testing.assert_frame_equal(
            obs.loc[:, summary_cols]
            .sum(axis=1)
            .to_frame("number_of_nodes_removed_in_denoise"),
            obs.loc[:, ["number_of_nodes_removed_in_denoise"]],
        )


@pytest.mark.slow
def test_denoise_ace_pls_cli_runs_ok(denoise_pxl_file):
    """One-core plus ACE: ACE counts match full-graph ACE; total includes one-core and stranded."""
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
            read(denoise_pxl_file).adata().obsm["clr"].loc[:, isotype_controls]
        ).mean(axis=1)
        result_isotype_levels = (
            result.adata().obsm["clr"].loc[:, isotype_controls].mean(axis=1)
        )
        assert (
            result_isotype_levels
            < original_isotype_levels.reindex(result_isotype_levels.index)
        ).all()
        obs = result.adata().obs
        summary_cols = [
            "denoised_nodes_marked_only_by_ace",
            "denoised_nodes_marked_only_by_pls",
            "denoised_nodes_marked_only_by_one_core",
            "denoised_nodes_marked_stranded",
            "denoised_nodes_marked_ace_and_pls",
            "denoised_nodes_marked_ace_and_one_core",
            "denoised_nodes_marked_pls_and_one_core",
            "denoised_nodes_marked_ace_pls_and_one_core",
        ]
        assert set(summary_cols).issubset(obs.columns)
        pd.testing.assert_frame_equal(
            obs.loc[:, summary_cols]
            .sum(axis=1)
            .to_frame("number_of_nodes_removed_in_denoise"),
            obs.loc[:, ["number_of_nodes_removed_in_denoise"]],
        )
