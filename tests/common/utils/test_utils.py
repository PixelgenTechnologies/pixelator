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
from pixelator.common.utils import cli as common_cli
from pixelator.common.utils import (
    get_available_cpu_count,
    get_part_number,
    get_pool_executor,
    get_process_pool_executor,
    get_read_sample_name,
    get_sample_name,
    gz_size,
    is_read_file,
    log_step_start,
    sanity_check_inputs,
    strip_sequence_file_suffixes,
)

# Every filename template that a pixelator stage writes, relative to a sample name.
STAGE_OUTPUT_TEMPLATES = [
    "{sample}.amplicon.fq.zst",
    "{sample}.report.json",
    "{sample}.meta.json",
    "{sample}.failed.fq.zst",
    "{sample}.amplicon.post_failed.fq.zst",
    "{sample}.demux.passed.fq.zst",
    "{sample}.demux.failed.fq.zst",
    "{sample}.demux.part_000.parquet",
    "{sample}.demux.m1.part_012.parquet",
    "{sample}.demux.m2.part_012.parquet",
    "{sample}.part_000.collapsed.parquet",
    "{sample}.collapsed.parquet",
    "{sample}.collapse.m1.part_010.parquet",
    "{sample}.collapse.m2.part_010.parquet",
    "{sample}.collapse.parquet",
    "{sample}.parquet",
    "{sample}.graph.pxl",
    "{sample}.denoised_graph.pxl",
    "{sample}.analysis.pxl",
    "{sample}.layout.pxl",
    "{sample}.dehashed.pxl",
    "{sample}.sample_calling.report.json",
]


def test_gzfile_is_empty(data_root):
    assert gz_size(data_root / "test_data_empty.fastq.gz") == 0


def test_gzfile_not_empty(data_root):
    assert gz_size(data_root / "test_data.fastq.gz") == 30858550


def test_gzfile_not_gz(tmp_path):
    not_gz = tmp_path / "not_gzip.txt"
    not_gz.write_text("this is not a gzip file\n")
    with pytest.raises(BadGzipFile):
        gz_size(not_gz)


def test_log_step_start(caplog):
    with caplog.at_level(logging.INFO):
        log_step_start(
            "my_step",
            input_files=["/foo", "/bar"],
            output="/fizz",
            a_param="hello",
            b_param="world",
        )
        # Only consider records emitted by log_step_start's own logger. Other
        # tests in the suite leave extra logging handlers attached to the
        # "pixelator" logger hierarchy, which makes caplog capture each record
        # once per attached logger. Collapse those duplicates while preserving
        # order so the assertion checks the logical sequence of emitted lines.
        messages = list(
            dict.fromkeys(
                rec.getMessage()
                for rec in caplog.records
                if rec.name == common_cli.logger.name
            )
        )
        assert messages == [
            f"Start pixelator my_step {__version__}",
            "Input file(s) /foo,/bar",
            "Output /fizz",
            "Parameters:a-param=hello,b-param=world",
        ]


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


@pytest.mark.parametrize("template", STAGE_OUTPUT_TEMPLATES)
@pytest.mark.parametrize(
    "sample",
    [
        "sample1",
        # A dot in the sample name must survive every stage suffix
        "QC2504_sample01_downsample_0.365_S1",
        "PNA055_Sample07_filtered_S7",
        "sample.with.several.dots",
    ],
)
def test_get_sample_name_round_trips_stage_outputs(sample, template):
    assert get_sample_name(template.format(sample=sample)) == sample


def test_get_sample_name_ignores_directories():
    assert get_sample_name("/tmp/some.dir/sample_0.365.graph.pxl") == "sample_0.365"


def test_get_sample_name_keeps_unknown_suffixes():
    # Only suffixes pixelator itself appends are stripped
    assert get_sample_name("sample_S1_L001") == "sample_S1_L001"
    assert get_sample_name("sample_R1.fastq.gz") == "sample_R1"
    assert get_sample_name("sample.v2.txt") == "sample.v2.txt"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sample1.demux.part_000.parquet", 0),
        ("sample1.demux.m1.part_012.parquet", 12),
        ("sample1.part_003", 3),
        ("QC2504_sample01_0.365_S1.demux.part_007.parquet", 7),
        # The last part suffix wins, as with the stage prefixes pixelator writes
        ("sample1.part_001.collapse.part_002.parquet", 2),
        ("sample1.demux.parquet", None),
        ("sample1.demux.part_abc.parquet", None),
        # A part suffix on a directory is not part of the file name
        ("/tmp/run.part_005/sample1.demux.parquet", None),
    ],
)
def test_get_part_number(name, expected):
    assert get_part_number(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Only sequence formats and compression are stripped, stage suffixes remain
        ("sample1.demux.passed.fq.zst", "sample1.demux.passed"),
        ("sample1.amplicon.fq.zst", "sample1.amplicon"),
        ("sample1_R1.fastq.gz", "sample1_R1"),
        ("sample1.parquet", "sample1.parquet"),
        (
            "QC2504_sample01_0.365_S1.demux.passed.fq.zst",
            "QC2504_sample01_0.365_S1.demux.passed",
        ),
        ("QC2504_sample01_0.365_S1_R1.fastq.gz", "QC2504_sample01_0.365_S1_R1"),
    ],
)
def test_strip_sequence_file_suffixes(name, expected):
    assert strip_sequence_file_suffixes(name) == expected


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


@pytest.mark.parametrize(
    "extension", ["fastq.gz", "fq.gz", "fastq.zst", "fq.zst", "fastq", "fq"]
)
def test_get_read_sample_name_with_dots_in_sample_name(extension):
    sample = "QC2504_sample01_downsample_0.365_S1"
    assert get_read_sample_name(f"{sample}_R1.{extension}") == sample
    assert get_read_sample_name(f"{sample}_R2.{extension}") == sample


@pytest.mark.parametrize(
    "extension", ["fastq.gz", "fq.gz", "fastq.zst", "fq.zst", "fastq", "fq"]
)
def test_is_read_file_with_dots_in_sample_name(extension):
    sample = "QC2504_sample01_downsample_0.365_S1"
    assert is_read_file(f"{sample}_R1.{extension}", read_type="r1")
    assert not is_read_file(f"{sample}_R1.{extension}", read_type="r2")
    assert is_read_file(f"{sample}_R2.{extension}", read_type="r2")
    assert not is_read_file(f"{sample}_R2.{extension}", read_type="r1")


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
    assert executor._max_workers == get_available_cpu_count()
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
    assert pool._processes == get_available_cpu_count()
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

    with patch("pixelator.common.utils.parallel.click") as click:
        click.get_current_context.return_value = MockContext()
        pool = get_pool_executor()
        assert isinstance(pool, Pool)
        assert pool._processes == 3
        assert pool._ctx == multiprocessing.get_context("spawn")
