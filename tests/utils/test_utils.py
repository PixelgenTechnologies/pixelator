"""
Tests for utility functions for the pixelator package

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import logging
import tempfile
from concurrent.futures import ProcessPoolExecutor
from gzip import BadGzipFile

import pytest

from pixelator import __version__
from pixelator.cli.logging import LoggingSetup
from pixelator.utils import (
    get_read_sample_name,
    gz_size,
    log_step_start,
    sanity_check_inputs,
    timer,
)


def test_gzfile_is_empty(data_root):
    assert gz_size(data_root / "test_data_empty.fastq.gz") == 0


def test_gzfile_not_empty(data_root):
    assert gz_size(data_root / "test_data.fastq.gz") == 30858550


def test_gzfile_not_gz(data_root):
    with pytest.raises(BadGzipFile):
        gz_size(data_root / "UNO_D21_Beta.csv")


def test_log_step_start(caplog):
    with caplog.at_level(logging.INFO):
        log_step_start(
            "my_step",
            input_files=["/foo", "/bar"],
            output="/fizz",
            a_param="hello",
            b_param="world",
        )
        records = [rec.message for rec in caplog.records]
        assert len(records) == 4
        assert records[0] == f"Start pixelator my_step {__version__}"
        assert records[1] == "Input file(s) /foo,/bar"
        assert records[2] == "Output /fizz"
        assert records[3] == "Parameters:a-param=hello,b-param=world"


def test_sanity_check_inputs_all_ok(data_root):
    sanity_check_inputs(
        input_files=[data_root / "test_data_R1.fastq.gz"],
        allowed_extensions="fastq.gz",
    )

    sanity_check_inputs(
        input_files=[data_root / "test_data_R1.fastq.gz"],
        allowed_extensions=("fq.gz", "fastq.gz"),
    )

    sanity_check_inputs(
        input_files=[data_root / "test_data.merged.fastq.gz"],
        allowed_extensions=("fq.gz", "fastq.gz"),
    )


def test_sanity_check_inputs_failed_criteria(data_root):
    with pytest.raises(AssertionError):
        sanity_check_inputs(
            input_files=[data_root / "test_data_R1.fastq.gz"],
            allowed_extensions="cat",
        )
    with pytest.raises(AssertionError):
        sanity_check_inputs(
            input_files=[data_root / "test_data_R3.fastq.gz"],
            allowed_extensions="fastq.gz",
        )
    with pytest.raises(AssertionError):
        sanity_check_inputs(
            input_files=[data_root / "test_data_R3.fastq.gz"],
            allowed_extensions=("csv", "txt"),
        )


def test_timer(caplog):
    @timer
    def my_func():
        return "foo"

    with caplog.at_level(logging.INFO):
        res = my_func()
        assert res == "foo"
        assert "Finished pixelator my_func in" in caplog.text


def test_verbose_logging_is_activated():
    test_log_file = tempfile.NamedTemporaryFile()
    with LoggingSetup(test_log_file.name, verbose=True):
        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.DEBUG


def test_verbose_logging_is_deactivated():
    test_log_file = tempfile.NamedTemporaryFile()
    with LoggingSetup(test_log_file.name, verbose=False):
        root_logger = logging.getLogger()
        assert root_logger.getEffectiveLevel() == logging.INFO


def helper_log_fn(args):
    import logging

    root_logger = logging.getLogger()
    root_logger.log(*args)


@pytest.mark.parametrize("verbose", [True, False])
def test_multiprocess_logging(verbose):
    """Test that logging works in a multiprocess environment."""
    test_log_file = tempfile.NamedTemporaryFile()

    with LoggingSetup(test_log_file.name, verbose=verbose):
        tasks = [
            (logging.DEBUG, "This is a debug message"),
            (logging.INFO, "This is an info message"),
            (logging.WARNING, "This is a warning message"),
            (logging.ERROR, "This is an error message"),
            (logging.CRITICAL, "This is a critical message"),
        ]

        with ProcessPoolExecutor(max_workers=4) as executor:
            for r in executor.map(helper_log_fn, tasks):
                pass

    # Test that the console output has the expected messages
    with open(test_log_file.name, "r") as f:
        log_content = f.read()

        if verbose:
            assert "This is a debug message" in log_content

        assert "This is an info message" in log_content
        assert "This is a warning message" in log_content
        assert "This is an error message" in log_content
        assert "This is a critical message" in log_content


def test_get_read_sample_name():
    with pytest.raises(ValueError, match="Invalid file extension.*"):
        get_read_sample_name("qwdwqwdqwd")

    with pytest.raises(ValueError, match="Invalid R1/R2 suffix."):
        get_read_sample_name("qwdwqwdqwd.fq.gz")

    assert get_read_sample_name("sample1_1.fq.gz") == "sample1"
    assert get_read_sample_name("sample1_2.fq.gz") == "sample1"

    assert get_read_sample_name("sample1_R1.fq.gz") == "sample1"
    assert get_read_sample_name("sample1_R2.fq.gz") == "sample1"
    assert get_read_sample_name("sample1_r1.fq.gz") == "sample1"
    assert get_read_sample_name("sample1_r2.fq.gz") == "sample1"
    assert get_read_sample_name("sample1.r1.fq.gz") == "sample1"
    assert get_read_sample_name("sample1.r2.fq.gz") == "sample1"
    assert get_read_sample_name("sample1.R1.fq.gz") == "sample1"
    assert get_read_sample_name("sample1.R2.fq.gz") == "sample1"

    assert get_read_sample_name("sample1_R1_L001.fq.gz") == "sample1_L001"
    assert get_read_sample_name("sample1_R2_L001.fq.gz") == "sample1_L001"
