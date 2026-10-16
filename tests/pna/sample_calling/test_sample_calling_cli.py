"""Tests for the sample calling CLI.

Copyright © 2025 Pixelgen Technologies AB.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from pixelator import cli
from pixelator.common.utils import get_sample_name
from pixelator.pna import read
from tests.pna.sample_calling.synthetic_data import (
    ENRICHMENT_THRESHOLD,
    N_CELLS,
    N_UNDETERMINED,
    POOL_NAME,
    SAMPLE_NAMES,
)


def test_runs_ok(synthetic_hashed_pxl, synthetic_samplesheet):
    """Verify runs ok.

    Args:
        synthetic_hashed_pxl: Synthetic hashed pixel file.
        synthetic_samplesheet: Samplesheet of the synthetic pool.
    """
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        pool_name = POOL_NAME
        args = [
            "single-cell-pna",
            "sample-calling",
            str(synthetic_hashed_pxl),
            "--samplesheet",
            str(synthetic_samplesheet),
            "--remove-incompatible",
            "--enrichment-threshold",
            str(ENRICHMENT_THRESHOLD),
            "--save-undetermined",
            "--output",
            d,
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        out_dir = Path(d) / "sample_calling"
        outputs = list(sorted(out_dir.glob("*.dehashed.pxl")))
        # One per sample + one for undetermined
        assert len(outputs) == len(SAMPLE_NAMES) + 1
        assert f"{pool_name}_undetermined.dehashed.pxl" in [o.name for o in outputs]

        for pxl_path in outputs:
            sample_name = get_sample_name(pxl_path)
            report_json = out_dir / f"{sample_name}.report.json"
            meta_json = out_dir / f"{sample_name}.meta.json"
            assert report_json.is_file(), f"missing per-sample report: {report_json}"
            assert meta_json.is_file(), f"missing per-sample meta: {meta_json}"

            report_data = json.loads(report_json.read_text())
            assert report_data["report_type"] == "sample_calling"
            assert report_data["sample_id"] == sample_name
            assert "number_of_components" in report_data
            assert report_data["input_reads"] > 0
            assert report_data["output_reads"] > 0
            assert report_data["output_reads"] <= report_data["input_reads"]

            meta_data = json.loads(meta_json.read_text())
            assert (
                meta_data["cli"]["command"]
                == "pixelator single-cell-pna sample-calling"
            )

        total_report = out_dir / f"{pool_name}.sample_calling.report.json"
        assert total_report.is_file(), "missing aggregated sample_calling.report.json"

        total_data = json.loads(total_report.read_text())
        assert total_data["report_type"] == "sample_calling_total"
        assert total_data["sample_id"] == "all"
        assert "hash_enrichment_factors_per_sample" in total_data
        assert "percentage_of_components_successfully_called" in total_data
        input_pxl = read(str(synthetic_hashed_pxl))
        expected_input_reads = int(input_pxl.adata().obs["reads_in_component"].sum())
        assert total_data["input_reads"] == expected_input_reads

        expected_output_reads = 0
        for output in outputs:
            if get_sample_name(output) == f"{pool_name}_undetermined":
                continue
            expected_output_reads += int(
                read(output).adata().obs["reads_in_component"].sum()
            )
        assert total_data["output_reads"] == expected_output_reads
        assert total_data["output_reads"] < total_data["input_reads"]
        assert (
            np.abs(
                total_data["percentage_of_components_successfully_called"]
                - (N_CELLS - N_UNDETERMINED) / N_CELLS
            )
            < 1e-6
        )
        total_components = 0
        for output in outputs:
            pxl = read(output)
            total_components += pxl.adata().obs.shape[0]
        assert total_components == N_CELLS
