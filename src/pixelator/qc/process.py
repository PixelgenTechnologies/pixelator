"""
This module contains functions for processing and QC any MPX amplicon design

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import logging
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

# List is used as type hint in comment
from typing import List  # noqa: F401

logger = logging.getLogger(__name__)


def qc_fastq(
    input: str,
    output: str,
    failed: str,
    report: str,
    metrics: str,
    design: str,
    n_limit: int,
    trim_front: int,
    trim_tail: int,
    min_length: int,
    max_length: int,
    threads: int,
    avg_qual: int,
    dedup: bool,
    remove_polyg: bool,
    verbose: bool,
) -> None:
    """
    This function is a wrapper around `fastp` to pre-process a `fastq`
    file. Duplicated sequences are removed, polyG sequences of length >= 5
    are trimmed. Some filters are used to discard reads (maximum and minimum
    read length and the maximum number of Ns in a read).

    :param input: the path to the fastq file
    :param output: the path to the output file (processed)
    :param failed: the path to the failed file (discarded reads)
    :param report: the path to the html report file
    :param metrics: the path to the json metrics file
    :param design: the design used in the config file
    :param n_limit: the number of Ns to use a cutoff to discard a read
    :param trim_front: the number of bases to trim in the front
    :param trim_tail: the number of bases to trim in the tail
    :param min_length: the minimum length for the reads
    :param max_length: the maximum length for the reads
    :param threads: the number of threads to use
    :param avg_qual: the minimum avg quality of a read
    :param dedup: remove duplicated reads when true
    :param remove_polyg: remove PolyG sequences (length 10 or more)
    :param verbose: run in verbose mode when true
    :returns: None
    :raises ValueError: raises an exception
            OSError: raises an exception
            CalledProcessError: raises an exception
            RuntimeError: raises an exception
    """
    args = [
        "fastp",
        "--disable_adapter_trimming",
        "--overrepresentation_analysis",
        "--thread",
        str(threads),
        "--n_base_limit",
        str(n_limit),
        "--trim_front1",
        str(trim_front),
        "--trim_tail1",
        str(trim_tail),
        "--max_len1",
        str(max_length),
        "--length_required",
        str(min_length),
        "--failed_out",
        failed,
        "--html",
        report,
        "--json",
        metrics,
        "--out1",
        output,
        "--average_qual",
        str(avg_qual),
        "--in1",
        input,
    ]  # type: List[str]

    if remove_polyg:
        args += ["--trim_poly_g", "--poly_g_min_len", "10"]
    if dedup:
        args += ["--dedup"]
    if verbose:
        args += ["--verbose"]

    logger.debug("Invoking fastp: %s", " ".join(args))

    try:
        proc = subprocess.Popen(  # type: ignore
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            shell=False,
        )
        (stdout, errmsg) = proc.communicate()
    except ValueError as exc:
        logger.error("ERROR running fastp. Incorrect arguments")
        raise exc
    except OSError as exc:
        logger.error("ERROR running fastp. Executable not found")
        raise exc
    except CalledProcessError as exc:
        logger.error("ERROR running fastp. Program returned error")
        raise exc

    error_string = errmsg.decode("utf-8")

    # check output file exists
    if not Path(output).is_file():
        error = (
            f"ERROR running fastp.\nOutput file not present "
            f"{output}\n{error_string}\n"
        )
        logger.error(error)
        raise RuntimeError(error)

    # Propagate unexpected fastp errors
    if proc.returncode != 0:
        msg = f"Fastp encountered an error: \n\n {error_string} \n"
        logger.error(msg)
        raise RuntimeError(msg)


def adapter_qc_fastq(
    input: str,
    output: str,
    failed: str,
    report: str,
    mismatches: float,
    pbs1: str,
    pbs2: str,
    cores: int,
    verbose: bool,
) -> None:
    """
    This function is a wrapper around `cutadapt` to process a `fastq`
    file with molecular pixelation data. The provided PBS1/2 sequences
    will be searched in the reads and only the reads that contain both
    sequences will be kept.

    :param input: the path to the fastq file (must contain PBS1/2 sequences)
    :param output: the path to the output file (processed)
    :param failed: the path to the failed file (discarded)
    :param report: the path to the json report
    :param mismatches: the number of mismatches allowed (0.1 - 0.9)
    :param min_length: the minimum length allowed in the PBS1/2 (match)
    :param pbs1: the PBS1 sequence
    :param pbs2: the PBS2 sequence
    :param cores: the number of threads to use
    :param verbose: run in verbose mode when true
    :returns: None
    :raises ValueError: raises an exception
            OSError: raises an exception
            CalledProcessError: raises an exception
            RuntimeError: raises an exception
    """
    min_overlap_pbs1 = len(pbs1) - int(mismatches * len(pbs1))
    min_overlap_pbs2 = len(pbs2) - int(mismatches * len(pbs2))
    args = [
        "cutadapt",
        "-e",
        str(mismatches),
        "--adapter",
        f"{pbs2};required;min_overlap={min_overlap_pbs2}...{pbs1};required;min_overlap={min_overlap_pbs1}",
        "--cores",
        str(cores),
        "--action=none",
        "--no-indels",
        "--untrimmed-output",
        failed,
        "--json",
        report,
        "--output",
        output,
        input,
    ]  # type: List[str]

    if verbose:
        args += ["--debug"]

    logger.debug("Invoking cutadapt: %s", " ".join(args))

    try:
        proc = subprocess.Popen(  # type: ignore
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            shell=False,
        )
        (stdout, errmsg) = proc.communicate()
    except ValueError as exc:
        logger.error("ERROR running cutadapt. Incorrect arguments")
        raise exc
    except OSError as exc:
        logger.error("ERROR running cutadapt. Executable not found")
        raise exc
    except CalledProcessError as exc:
        logger.error("ERROR running cutadapt. Program returned error")
        raise exc

    error_string = errmsg.decode("utf-8")

    # check output file exists

    if not Path(output).is_file():
        error = (
            f"ERROR running cutadapt.\nOutput file not present "
            f"{output}\n{stdout.decode('utf-8')}\n{errmsg.decode('utf-8')}"
        )
        logger.error(error)
        raise RuntimeError(error)

    # Propagate unexpected cutadapt errors
    if proc.returncode != 0:
        msg = f"Cutadapt encountered an error: \n\n {error_string} \n"
        logger.error(msg)
        raise RuntimeError(msg)
