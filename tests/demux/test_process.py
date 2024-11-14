"""Copyright © 2024 Pixelgen Technologies AB."""

from pixelator.demux.process import check_demux_results_are_ok


def test_check_demux_results_are_ok():
    data = {"read_counts": {"input": 100, "output": 90}}
    assert check_demux_results_are_ok(data, sample_id="test")


def test_check_demux_results_are_not_ok():
    data = {"read_counts": {"input": 100, "output": 40}}
    assert not check_demux_results_are_ok(data, sample_id="test")
