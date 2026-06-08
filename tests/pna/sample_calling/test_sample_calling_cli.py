"""Tests for the sample calling CLI.

Copyright © 2025 Pixelgen Technologies AB.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from pixelator import cli
from pixelator.common.utils import get_sample_name
from pixelator.pna import read


@pytest.mark.slow
def test_runs_ok(pna_data_root):
    """Verify runs ok.

    Args:
        pna_data_root: pna data root.
    """
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as d:
        pool_name = "small_hashed_sample_1"
        args = [
            "single-cell-pna",
            "sample-calling",
            str(pna_data_root / f"sample_calling/{pool_name}.pxl"),
            "--samplesheet",
            str(pna_data_root) + "/sample_calling/samplesheet.csv",
            "--remove-incompatible",
            "--confidence-threshold",
            "0.96",
            "--save-undetermined",
            "--output",
            d,
        ]
        cmd = runner.invoke(cli.main_cli, args)

        assert cmd.exit_code == 0

        out_dir = Path(d) / "sample_calling"
        outputs = list(sorted(out_dir.glob("*.dehashed.pxl")))
        # One per sample + one for undetermined
        assert len(outputs) == 4
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
        assert "sample_confidences_per_sample" in total_data
        assert "percentage_of_components_successfully_called" in total_data
        assert (
            np.abs(total_data["percentage_of_components_successfully_called"] - 14 / 15)
            < 1e-6
        )
        total_components = 0
        for output in outputs:
            pxl = read(output)
            total_components += pxl.adata().obs.shape[0]
        assert total_components == 15
