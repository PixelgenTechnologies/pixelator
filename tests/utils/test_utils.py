"""
Tests for utility functions for the pixelator package

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import logging
from gzip import BadGzipFile
from tempfile import NamedTemporaryFile

import pytest

from pixelator import __version__
from pixelator.cli.common import init_logger
from pixelator.utils import (
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
    with NamedTemporaryFile() as test_log_file:
        init_logger(log_file=test_log_file.name, verbose=True)

        root_logger = logging.getLogger("pixelator")
        assert root_logger.getEffectiveLevel() == logging.DEBUG

        init_logger(log_file=test_log_file.name, verbose=False)
        assert root_logger.getEffectiveLevel() == logging.INFO
