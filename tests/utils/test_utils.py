"""
Tests for utility functions for the pixelator package

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from gzip import BadGzipFile
from multiprocessing.pool import Pool
from unittest.mock import patch

import pytest

from pixelator import __version__
from pixelator.utils import (
    get_pool_executor,
    get_process_pool_executor,
    get_read_sample_name,
    gz_size,
    is_read_file,
    log_step_start,
    sanity_check_inputs,
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


def test_sanity_check_inputs_single_file_ok(data_root):
    sanity_check_inputs(
        input_files=data_root / "test_data_R1.fastq.gz",
        allowed_extensions="fastq.gz",
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

    assert get_read_sample_name("sample1_L001_R1.fq.gz") == "sample1_L001"
    assert get_read_sample_name("sample1_L001_R2.fq.gz") == "sample1_L001"

    # Check that illumina numbered suffixes are recognised and remain
    # in the sample name
    assert get_read_sample_name("sample1_L001_R1_001.fq.gz") == "sample1_L001_001"
    assert get_read_sample_name("sample1_L001_R2_001.fq.gz") == "sample1_L001_001"

    # Check that the right `_1` is removed when there are multiple matches
    assert get_read_sample_name("sample_ABCD_12345_1.fastq.gz") == "sample_ABCD_12345"
    # Check that the right `_2` is removed when there are multiple matches
    assert (
        get_read_sample_name("sample_ABCD_2234_2___2_66_2.fastq.gz")
        == "sample_ABCD_2234_2___2_66"
    )


def test_is_read_file():
    with pytest.raises(ValueError, match="Invalid file extension.*"):
        is_read_file("qwdwqwdqwd", read_type="r1")

    with pytest.raises(
        AssertionError, match="Invalid read type: expected 'r1' or 'r2'"
    ):
        is_read_file("sample1.r1.fq.gz", read_type="qdqwdqw")

    for r1_check in [
        "sample1_1.fq.gz",
        "sample1_R1.fq.gz",
        "sample1_r1.fq.gz",
        "sample1.r1.fq.gz",
        "sample1.R1.fq.gz",
        "sample_1_R2_R1.fq.gz",
    ]:
        assert is_read_file(r1_check, read_type="r1")
        assert not is_read_file(r1_check, read_type="r2")

    for r2_check in [
        "sample1_2.fq.gz",
        "sample1_R2.fq.gz",
        "sample1_r2.fq.gz",
        "sample1.r2.fq.gz",
        "sample1.R2.fq.gz",
        "sample_R1_2_R2.fq.gz",
    ]:
        assert is_read_file(r2_check, read_type="r2")
        assert not is_read_file(r2_check, read_type="r1")

    # Check that read suffixes are only tested at the end of the file name
    assert is_read_file("sample_1_dwwdwdw_R1.fq.gz", read_type="r1")

    # Check that read suffixes are not checked in path components
    assert is_read_file("sample_R2/sample_1_dwwdwdw_R1.fq.gz", read_type="r1")
    assert is_read_file("sample_R1/sample_1_dwwdwdw_R2.fq.gz", read_type="r2")

    # Check that illumina numbered suffixes are recognised
    assert is_read_file("sample_1_dwwdwdw_R1_001.fq.gz", read_type="r1")
    assert not is_read_file("sample_1_dwwdwdw_R1_001.fq.gz", read_type="r2")

    assert is_read_file("sample_1_dwwdwdw_R2_003.fq.gz", read_type="r2")
    assert not is_read_file("sample_1_dwwdwdw_R2_003.fq.gz", read_type="r1")


def test_is_read_file_should_be_ok_when_r1_or_r2_in_dir_name():
    # not the r1 in the directory name
    file_name = "/tmp/tmp5r1eg53r/uropod_control_R1.fastq.gz"
    assert is_read_file(file_name, "r1")

    # not the r2 in the directory name
    file_name = "/tmp/tmp5r2eg53r/uropod_control_R1.fastq.gz"
    assert is_read_file(file_name, "r1")


def test_get_process_pool_executor():
    # Test with default parameters
    executor = get_process_pool_executor()
    assert isinstance(executor, ProcessPoolExecutor)
    assert executor._max_workers == multiprocessing.cpu_count()
    assert executor._mp_context == multiprocessing.get_context("spawn")

    # Test with specified number of cores
    executor = get_process_pool_executor(nbr_cores=4)
    assert isinstance(executor, ProcessPoolExecutor)
    assert executor._max_workers == 4
    assert executor._mp_context == multiprocessing.get_context("spawn")

    # Test set context
    executor = get_process_pool_executor(nbr_cores=2, context="fork")
    assert isinstance(executor, ProcessPoolExecutor)
    assert executor._max_workers == 2
    assert executor._mp_context == multiprocessing.get_context("fork")


def test_get_pool_executor():
    # Test with default parameters
    pool = get_pool_executor()
    assert isinstance(pool, Pool)
    assert pool._processes == multiprocessing.cpu_count()
    assert pool._ctx == multiprocessing.get_context("spawn")

    # Test with specified number of cores
    pool = get_pool_executor(nbr_cores=4)
    assert isinstance(pool, Pool)
    assert pool._processes == 4
    assert pool._ctx == multiprocessing.get_context("spawn")

    # Test set context
    pool = get_pool_executor(nbr_cores=4, context="fork")
    assert isinstance(pool, Pool)
    assert pool._processes == 4
    assert pool._ctx == multiprocessing.get_context("fork")


def test_get_pool_executor_with_click_context():
    class MockContext:
        @property
        def obj(self):
            return {"CORES": 3}

    with patch("pixelator.utils.click") as click:
        click.get_current_context.return_value = MockContext()
        pool = get_pool_executor()
        assert isinstance(pool, Pool)
        assert pool._processes == 3
        assert pool._ctx == multiprocessing.get_context("spawn")
