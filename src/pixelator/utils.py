"""
Common functions and utilities for Pixelator

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from __future__ import annotations

import collections.abc
import gzip
import itertools
import json
import logging
import tempfile
import textwrap
import time
from functools import wraps
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Sequence, Set, TYPE_CHECKING, Union

import click
import numpy as np
import pandas as pd

from pixelator import __version__
from pixelator.types import PathType

# Avoid a circular dependency
if TYPE_CHECKING:
    from pixelator.config import AntibodyPanel

logger = logging.getLogger(__name__)

# this tr table is used to complement DNA sequences
_TRTABLE = str.maketrans("GTACN", "CATGN")


def build_barcodes_file(
    panel: AntibodyPanel, anchored: bool, rev_complement: bool
) -> str:
    """
    Utility function to create a FASTA file of barcodes from a
    panel dataframe. The FASTA file will have the marker id as
    name and the barcode sequence as sequence. The parameter
    rev_complement control if sequence needs to be in reverse
    complement form or not. When anchored is true a dollar sign
    ($) will be added at the end of the sequences to make them
    anchored in cutadapt.

    :param panel: an Antibody panel object
    :param anchored: make the sequences anchored if True
    :param rev_complement: reverse complement the sequence column
                           if True
    :returns: a path to the barcodes file
    """
    logger.debug("Creating barcodes file from antibody panel")

    fd = tempfile.NamedTemporaryFile(suffix=".fa", delete=False)
    logger.info("Barcodes file saved in %s", fd.name)
    with open(fd.name, "w") as fh:
        for _, row in panel.df.iterrows():
            marker_id = row["marker_id"]
            sequence = row["sequence"]
            fh.write(f">{sequence} [marker_id={marker_id}]\n")
            seq = reverse_complement(sequence) if rev_complement else sequence
            if anchored:
                seq += "$"
            fh.write(f"{seq}\n")

    logger.debug("Barcodes file from antibody panel created")
    return fd.name


def click_echo(msg: str, multiline: bool = False):
    """
    Helper function that print a line to the console
    with long-line wrapping.

    :param msg: the message to print
    :param multiline: True to use text wrapping or False otherwise (default)
    """
    if multiline:
        click.echo(textwrap.fill(textwrap.dedent(msg), width=100))
    else:
        click.echo(msg)


def create_output_stage_dir(root: PathType, name: str) -> Path:
    """
    Create a new subfolder with `name` under the given `root` directory.

    :param root: the root directory
    :returns: the created folder (Path)
    """
    output = Path(root) / name
    if not output.is_dir():
        output.mkdir(parents=True)
    return output


def flatten(list_of_collections: List[Union[List, Set]]) -> List:
    """
    Flattens a list of lists or list of sets.

    :param list_of_collections: list of lists or list of sets
    :returns: list containing flattened items
    """
    return [item for sublist in list_of_collections for item in sublist]


def get_extension(filename: PathType, len_ext: int = 2) -> str:
    """
    Utility function to extract file extensions.

    :param filename: the file name
    :param len: the extension length
    :returns: the file extension (str)
    """
    return "".join(PurePath(filename).suffixes[-len_ext:]).lstrip(".")


def get_sample_name(filename: PathType) -> str:
    """
    Extract the sample name from a sample's filename.
    The sample name is expected to be from the start of the filename until
    the first dot.

    :param filename: path to the file
    :returns: the sample name
    """
    return Path(filename).stem.split(".")[0]


def group_input_reads(
    inputs: Sequence[PathType], input1_pattern: str, input2_pattern: str
) -> Dict[str, List[Path]]:
    """
    Group input files by read pairs and sample id

    :param inputs: list of input files
    :param input1_pattern: pattern to match read1 files
    :param input2_pattern: pattern to match read2 files
    :raises ValueError: if the number of reads for a sample is more than 2
    :returns: a dictionary with the grouped reads
    """

    def group_fn(s):
        sn = get_sample_name(s)
        return sn.replace(input1_pattern, "").replace(input2_pattern, "")

    inputs = sorted(inputs, key=group_fn)
    # group reads by sample id
    grouped_inputs = {
        key: list(val_iter) for key, val_iter in itertools.groupby(inputs, group_fn)
    }

    # If the input contains 2 files, match them to the read1 and read2 patterns
    # otherwise, assume that the input is a single file and ignore the read patterns
    sorted_grouped_reads = {}
    for key, values in grouped_inputs.items():
        if len(values) == 2:
            input1 = sorted([Path(x) for x in values if input1_pattern in str(x)])
            input2 = sorted([Path(x) for x in values if input2_pattern in str(x)])

            if len(input1) != 1:
                raise click.ClickException(
                    f"Expected an input files identified with {input1_pattern}"
                )

            if len(input2) != 1:
                raise click.ClickException(
                    f"Expected an input files identified with {input2_pattern}"
                )

            sorted_grouped_reads[key] = [input1[0], input2[0]]
        elif len(values) == 1:
            sorted_grouped_reads[key] = [Path(values[0])]
        else:
            raise ValueError(f"Unexpected number of inputs for sample {key}")

    return sorted_grouped_reads


def gz_size(filename: str) -> int:
    """
    Extract the size of a gzip compressed file.

    :param fname: file name
    :returns: size of the file uncompressed (in bits)
    """
    with gzip.open(filename, "rb") as f:
        return f.seek(0, whence=2)


def log_step_start(
    step_name: str,
    input_files: Optional[List[str]] = None,
    output: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Utility function to add information about the start of a
    pixelator step to the logs

    :param step_name: name of the step that is starting
    :param input_files: optional collection of input file paths
    :param output: optional path to output
    :param kwargs: any additional parameters that you wish to log
    :returns: None
    """
    logger.info("Start pixelator %s %s", step_name, __version__)

    if input_files is not None:
        logger.info("Input file(s) %s", ",".join(input_files))

    if output is not None:
        logger.info("Output %s", output)

    if kwargs is not None:
        params = [f"{key.replace('_', '-')}={value}" for key, value in kwargs.items()]
        logger.info("Parameters:%s", ",".join(params))


def np_encoder(object: Any):
    """
    A very simple encoder to allow JSON serialization
    of numpy data types
    """
    if isinstance(object, np.generic):
        return object.item()


def remove_csv_whitespaces(df: pd.DataFrame) -> None:
    """
    Utility function to remove leading and trailing
    blank spaces from csv files slurped by pandas
    """
    # fill NaNs as empty strings to be able to do `.str`
    df.fillna("", inplace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].str.strip()


def reverse_complement(seq: str) -> str:
    """
    Helper function to compute the reverse complement of a DNA seq

    :param seq: the DNA sequence
    :returns: the reverse complement of the input sequence
    """
    return seq.translate(_TRTABLE)[::-1]


def sanity_check_inputs(
    input_files: Sequence[PathType],
    allowed_extensions: Union[Sequence[str], Optional[str]] = None,
) -> None:
    """
    Perform basic sanity checking of input files

    :param input_files: the files to sanity check
    :param allowed_extensions: the expected file extension of the files, e.g. 'fastq.gz'
                               or a tuple of allowed types eg. ('fastq.gz', 'fq.gz')
    :returns: None
    :raises AssertionError: when any of validation fails
    """
    for input_file in input_files:
        input_file = Path(input_file)
        logger.debug("Sanity checking %s", input_file)

        if not input_file.is_file():
            raise AssertionError(f"{input_file} is not a file")

        if input_file.stat().st_size == 0:
            raise AssertionError(f"{input_file} is an empty file")

        if not isinstance(allowed_extensions, str) and isinstance(
            allowed_extensions, collections.abc.Sequence
        ):
            if not any(str(input_file).endswith(ext) for ext in allowed_extensions):
                raise AssertionError(
                    f"{input_file} does not have any of the "
                    f"extensions {', '.join(allowed_extensions)}"
                )
        elif allowed_extensions is not None and not str(input_file).endswith(
            allowed_extensions
        ):
            raise AssertionError(
                f"{input_file} does not have the extension {allowed_extensions}"
            )


def single_value(xs: Union[List, Set]) -> Any:
    """
    Extract the first value in a List or Set if the
    collection has a single value.

    :param xs: a collection of values
    :returns: the first value in the collection
    :raises AssertionError: if the collection is empty or has more than one value
    """
    if len(xs) == 0:
        raise AssertionError("Empty collection")
    if len(xs) > 1:
        raise AssertionError("More than one element in collection")
    return list(xs)[0]


def timer(func):
    """
    Function decorator used to time the different steps
    """

    @wraps(func)
    def wrapper(*args, **kwds):
        start_time = time.perf_counter()
        res = func(*args, **kwds)
        run_time = time.perf_counter() - start_time
        logger.info("Finished pixelator %s in %.2fs", func.__name__, run_time)
        return res

    return wrapper


def write_parameters_file(
    click_context: click.Context, output_file: Path, command_path: Optional[str] = None
):
    """
    Write the parameters used in for a command to a JSON file

    :param click_context: the click context object
    :param output_file: the output file
    :param command_path: the command to use as command name
    :returns: None
    """
    command_path_fixed = command_path or click_context.command_path
    parameters = click_context.command.params
    parameter_values = click_context.params

    param_data = {}

    for param in parameters:
        if not isinstance(param, click.core.Option):
            continue

        name = param.opts[0]
        value = parameter_values.get(str(param.name))
        if value is not None and param.type == click.Path:
            value = str(Path(value).resolve())

        param_data[name] = value

    data = {
        "cli": {
            "command": command_path_fixed,
            "options": param_data,
        }
    }

    logger.debug("Writing parameters file to %s", str(output_file))

    with open(output_file, "w") as fh:
        json.dump(data, fh, indent=4)
