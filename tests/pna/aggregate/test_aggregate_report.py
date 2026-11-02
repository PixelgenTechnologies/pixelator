"""Tests for AggregateSampleReport.

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

from pixelator.pna.aggregate.report import AggregateSampleReport


def test_aggregate_sample_report_round_trip(tmp_path: Path):
    """Report serializes to JSON and reloads with computed output_reads."""
    report = AggregateSampleReport(
        product_id="single-cell-pna",
        sample_id="sample",
        input_reads=10,
        output_molecules=4,
        umi1_degree_distribution={1: 3, 2: 1},
        umi2_degree_distribution={1: 2, 3: 1},
    )

    assert report.report_type == "aggregate"
    assert report.output_reads == 10

    path = tmp_path / "sample.report.json"
    report.write_json_file(path, indent=4)

    loaded = AggregateSampleReport.from_json(path)
    assert loaded.report_type == "aggregate"
    assert loaded.sample_id == "sample"
    assert loaded.input_reads == 10
    assert loaded.output_reads == 10
    assert loaded.output_molecules == 4
    assert loaded.umi1_degree_distribution == {1: 3, 2: 1}
    assert loaded.umi2_degree_distribution == {1: 2, 3: 1}
