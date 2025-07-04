"""Module that contains functions for demultiplex reads belonging to antibody panels.

Copyright © 2022 Pixelgen Technologies AB.
"""

import json
import logging
import subprocess
from pathlib import Path
from subprocess import CalledProcessError

# List is used as type hint in comment
from typing import List  # noqa: F401

from pixelator.common.config.panel import AntibodyPanel

logger = logging.getLogger(__name__)

DEMUX_PERCENTAGE_THRESHOLD = 0.5


def demux_fastq(
    input: str,
    output: str,
    failed: str,
    report: str,
    panel: AntibodyPanel,
    mismatches: float,
    barcodes: str,
    min_length: int,
    cores: int,
    verbose: bool,
    sample_id: str,
) -> bool:
    """Demultiplex the input fastq file into separate files based on the barcodes.

    This function is a wrapper around `cutadapt` to process a `fastq`
    file with molecular pixelation data. A `fasta` file with the barcodes
    to use to demultiplex must be provided (antibody name and its sequence).
    The fastq will be processed to demultiplex its barcode with the ones
    provided in the fasta file (--barcodes). One fastq file per barcode
    will be generated (containing the reads that matched). Another single
    file with the reads that failed to demultiplex will also be created.

    :param input: the path to the fastq file (must contain the barcode)
    :param output: the path to the output file (processed)
    :param failed: the path to the failed file (discarded)
    :param report: the path to the json report
    :param panel: the path to the panel used for demultiplexing
    :param mismatches: the number of mismatches allowed (a percentage)
    :param barcodes: the fasta file containing the barcodes (reference)
    :param min_length: the minimum overlap required in the barcode
    :param cores: the number of cores to use
    :param verbose: run in verbose mode when true
    :param sample_id: the sample id
    :returns: true if demux results were ok
    :raises ValueError: raises an exception
            OSError: raises an exception
            CalledProcessError: raises an exception
            RuntimeError: raises an exception
    """
    args = [
        "cutadapt",
        "-e",
        str(mismatches),
        "--adapter",
        f"file:{barcodes}",
        "--cores",
        str(cores),
        "--action=none",
        "--no-indels",
        "--untrimmed-output",
        failed,
        "--overlap",
        str(min_length),
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

    # check output file exists
    if not Path(report).is_file():
        error = (
            f"ERROR running cutadapt.\nOutput file not present "
            f"{report}\n{stdout.decode('utf-8')}\n{errmsg.decode('utf-8')}"
        )
        logger.error(error)
        raise RuntimeError(error)

    modified_data = {}
    with open(report, "r") as fh:
        file_data = json.load(fh)

        for adapters in file_data["adapters_read1"]:
            adapters["name"] = panel.get_marker_id(adapters["name"])

        modified_data = file_data
        modified_data["sample_id"] = sample_id

    with open(report, "w") as fh:
        json.dump(modified_data, fh, indent=4)

    return check_demux_results_are_ok(modified_data, sample_id)


def check_demux_results_are_ok(report_data: dict, sample_id: str) -> bool:
    """Check if the demultiplexing results are ok."""
    read_counts = report_data["read_counts"]
    input_reads = read_counts["input"]
    output_reads = read_counts["output"]

    if output_reads / input_reads < DEMUX_PERCENTAGE_THRESHOLD:
        logger.error(
            (
                "Low demultiplexing rate (%.2f%%) for %s. Your results may not be reliable."
                "Are you sure you have used the correct panel file?"
            ),
            output_reads / input_reads * 100,
            sample_id,
        )
        return False
    return True
