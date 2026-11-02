"""Tests for EdgelistSampleReport.

Copyright © 2026 Pixelgen Technologies AB.
"""

import json
from pathlib import Path

from pixelator.pna.edgelist.report import EdgelistSampleReport


def _report(**overrides) -> EdgelistSampleReport:
    """An otherwise valid report, with fields replaceable per test."""
    fields = {
        "product_id": "single-cell-pna",
        "sample_id": "sample",
        "input_reads": 1000,
        "output_reads": 900,
        "q30_statistics": {
            "total": 0.94,
            "umi1": 0.97,
            "pid1": 0.98,
            "pid2": 0.96,
            "umi2": 0.95,
        },
        "dropped_by": {"pid1_not_in_panel": 60, "has_n": 40, "too_short": 0},
        "reads_per_marker_1": {"CD4": 500, "CD8": 400, "CD3": 0},
        "reads_per_marker_2": {"CD8": 600, "CD4": 300, "CD3": 0},
    }
    fields.update(overrides)
    return EdgelistSampleReport(**fields)


def test_report_type_identifies_the_new_edgelist_stage():
    """The stage replaces amplicon + demux but reports as `edgelist`."""
    assert _report().report_type == "edgelist"


def test_report_round_trips_through_json(tmp_path: Path):
    """Every field survives a write and reload unchanged."""
    report = _report()

    path = tmp_path / "sample.report.json"
    report.write_json_file(path, indent=4)

    loaded = EdgelistSampleReport.from_json(path)
    assert loaded.product_id == "single-cell-pna"
    assert loaded.report_type == "edgelist"
    assert loaded.sample_id == "sample"
    assert loaded.input_reads == 1000
    assert loaded.output_reads == 900
    assert loaded.q30_statistics.total == 0.94
    assert loaded.q30_statistics.umi1 == 0.97
    assert loaded.q30_statistics.pid1 == 0.98
    assert loaded.q30_statistics.pid2 == 0.96
    assert loaded.q30_statistics.umi2 == 0.95
    assert loaded.dropped_by == {"pid1_not_in_panel": 60, "has_n": 40, "too_short": 0}
    assert list(loaded.reads_per_marker_1.items()) == [
        ("CD4", 500),
        ("CD8", 400),
        ("CD3", 0),
    ]
    assert list(loaded.reads_per_marker_2.items()) == [
        ("CD8", 600),
        ("CD4", 300),
        ("CD3", 0),
    ]


def test_written_json_carries_the_fields_experiment_summary_reads(tmp_path: Path):
    """The on-disk shape is the contract, so assert it rather than the model.

    Reading the file back through pydantic would hide a renamed or nested key.
    """
    path = tmp_path / "sample.report.json"
    _report().write_json_file(path, indent=4)

    payload = json.loads(path.read_text())

    assert payload["sample_id"] == "sample"
    assert payload["product_id"] == "single-cell-pna"
    assert payload["report_type"] == "edgelist"
    assert payload["input_reads"] == 1000
    assert payload["output_reads"] == 900
    assert payload["q30_statistics"] == {
        "total": 0.94,
        "umi1": 0.97,
        "pid1": 0.98,
        "pid2": 0.96,
        "umi2": 0.95,
    }
    # Key order carries the ranking, so compare as lists.
    assert list(payload["reads_per_marker_1"].items()) == [
        ("CD4", 500),
        ("CD8", 400),
        ("CD3", 0),
    ]
    assert list(payload["reads_per_marker_2"].items()) == [
        ("CD8", 600),
        ("CD4", 300),
        ("CD3", 0),
    ]


def test_written_json_keeps_drop_reasons_in_the_order_given(tmp_path: Path):
    """Ranking is carried by key order, and only insertion order preserves it.

    Sorting anywhere along the write path would quietly reorder the reasons.
    """
    ranked = {
        "pid2_not_in_panel": 90,
        "has_n": 40,
        "pid1_not_in_panel": 40,
        "too_short": 0,
    }
    path = tmp_path / "sample.report.json"
    _report(dropped_by=ranked).write_json_file(path, indent=4)

    payload = json.loads(path.read_text())

    assert list(payload["dropped_by"]) == list(ranked)
