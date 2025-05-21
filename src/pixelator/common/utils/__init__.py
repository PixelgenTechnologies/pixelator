"""Common functions and utilities for Pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import collections.abc
import gzip
import itertools
import json
import logging
import multiprocessing
import re
import tempfile
import textwrap
import time
import typing
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from logging.handlers import SocketHandler
from multiprocessing.pool import Pool
from pathlib import Path, PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Union,
)

import click
import numpy as np
import pandas as pd

from pixelator.common.types import PathType

# Avoid a circular dependency
if TYPE_CHECKING:
    from pixelator.common.config import AntibodyPanel

logger = logging.getLogger(__name__)

# this tr table is used to complement DNA sequences
_TRTABLE = str.maketrans("GTACN", "CATGN")


def build_barcodes_file(
    panel: AntibodyPanel, anchored: bool, rev_complement: bool
) -> str:
    """Create a FASTA file of barcodes from a panel dataframe.

    The FASTA file will have the marker id as
    name and the barcode sequence as sequence. The parameter
    rev_complement control if sequence needs to be in reverse
    complement form or not. When anchored is true a dollar sign
    ($) will be added at the end of the sequences to make them
    anchored in cutadapt.

    :param panel: an Antibody panel object
    :param anchored: make the sequences anchored if True
    :param rev_complement: reverse complement the sequence column
                           if True
    :returns str: a path to the barcodes file
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
    """Print a line to the console with optional long-line wrapping.

    :param msg: the message to print
    :param multiline: True to use text wrapping or False otherwise (default)
    """
    if multiline:
        click.echo(textwrap.fill(textwrap.dedent(msg), width=100))
    else:
        click.echo(msg)


def create_output_stage_dir(root: PathType, name: str) -> Path:
    """Create a new subfolder with `name` under the given `root` directory.

    :param root: the parent directory
    :param name: the name of the directory to create
    :returns Path: the created folder (Path)
    """
    output = Path(root) / name
    if not output.is_dir():
        output.mkdir(parents=True)
    return output


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter.

    Taken from python itertools recipes.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def get_extension(filename: PathType, len_ext: int = 2) -> str:
    """Extract file extensions from a filename.

    :param filename: the file name
    :param len_ext: the number of expected extensions parts
        e.g.: fq.gz gives len_ext=2
    :returns str: the file extension (str)
    """
    return "".join(PurePath(filename).suffixes[-len_ext:]).lstrip(".")


def get_sample_name(filename: PathType) -> str:
    """Extract the sample name from a sample's filename.

    The sample name is expected to be from the start of the filename until
    the first dot.

    :param filename: path to the file
    :returns str: the sample name
    """
    return Path(filename).stem.split(".")[0]


def group_input_reads(
    inputs: Sequence[PathType], input1_pattern: str, input2_pattern: str
) -> Dict[str, List[Path]]:
    """Group input files by read pairs and sample id.

    :param inputs: list of input files
    :param input1_pattern: pattern to match read1 files
    :param input2_pattern: pattern to match read2 files
    :raises ValueError: if the number of reads for a sample is more than 2
    :returns Dict[str, List[Path]]: a dictionary with the grouped reads
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
    """Extract the size of a gzip compressed file.

    :param filename: file name
    :returns int: size of the file uncompressed (in bits)
    """
    with gzip.open(filename, "rb") as f:
        return f.seek(0, whence=2)


def log_step_start(
    step_name: str,
    input_files: Optional[List[str] | str] = None,
    output: Optional[str] = None,
    **kwargs,
) -> None:
    """Add information about the start of a pixelator step to the logs.

    :param step_name: name of the step that is starting
    :param input_files: collection of input file paths
    :param output: optional path to output
    :param **kwargs: any additional parameters that you wish to log
    :rtype: None
    """
    from pixelator import __version__

    logger.info("Start pixelator %s %s", step_name, __version__)

    if isinstance(input_files, list):
        logger.info("Input file(s) %s", ",".join(input_files))

    if isinstance(input_files, str):
        logger.info("Input file %s", input_files)

    if output is not None:
        logger.info("Output %s", output)

    if kwargs is not None:
        params = [f"{key.replace('_', '-')}={value}" for key, value in kwargs.items()]
        logger.info("Parameters:%s", ",".join(params))


def np_encoder(object: Any):
    """Encoder for JSON serialization of numpy data types."""  # noqa: D401
    if isinstance(object, np.generic):
        return object.item()


def remove_csv_whitespaces(df: pd.DataFrame) -> None:
    """Remove leading and trailing blank spaces from csv files slurped by pandas."""
    # fill NaNs as empty strings to be able to do `.str`
    df.fillna("", inplace=True)
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].str.strip()


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA seq.

    :param seq: the DNA sequence
    :return: the reverse complement of the input sequence
    :rtype: str
    """
    return seq.translate(_TRTABLE)[::-1]


def sanity_check_inputs(
    input_files: Sequence[PathType] | PathType,
    allowed_extensions: Union[Sequence[str], Optional[str]] = None,
) -> None:
    """Perform basic sanity checking of input files.

    :param input_files: the files to sanity check
    :param allowed_extensions: the expected file extension of the files, e.g. 'fastq.gz'
                               or a tuple of allowed types eg. ('fastq.gz', 'fq.gz')
    :raises AssertionError: when any of validation fails
    :returns None:
    """
    input_files_: list[PathType] = (
        input_files if not isinstance(input_files, PathType) else [input_files]  # type: ignore
    )

    for input_file in input_files_:
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


T = typing.TypeVar("T")


def single_value(xs: Union[List[T], Set[T]]) -> T:
    """Extract the first value in a List or Set if the collection has a single value.

    :param xs: a collection of values
    :returns T: the first value in the collection
    :raises AssertionError: if the collection is empty or has more than one value
    """
    if len(xs) == 0:
        raise AssertionError("Empty collection")
    if len(xs) > 1:
        raise AssertionError("More than one element in collection")
    return list(xs)[0]


def timer(func):
    """Time the different steps of a function."""

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
) -> None:
    """Write the parameters used in for a command to a JSON file.

    :param click_context: the click context object
    :param output_file: the output file
    :param command_path: the command to use as command name
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


def _add_handlers_to_root_logger(port, log_level):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    socket_handler = SocketHandler("localhost", port)
    root_logger.addHandler(socket_handler)


def _pre_multiprocessing_args(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
):
    # If these variable are not set we will try to pick them
    # up from the click context
    current_click_context = click.get_current_context(silent=True)
    click_logging_setup = None
    click_nbr_cores = None
    if current_click_context:
        click_logging_setup = current_click_context.obj.get("LOGGER")
        click_nbr_cores = current_click_context.obj.get("CORES")

    nbr_cores = (
        nbr_cores if nbr_cores else click_nbr_cores or multiprocessing.cpu_count()
    )
    args_dict = {
        "max_workers": nbr_cores,
        "mp_context": multiprocessing.get_context(context),
    }

    if logging_setup or click_logging_setup:
        args_dict = args_dict | dict(
            initializer=_add_handlers_to_root_logger,
            initargs=(
                (logging_setup or click_logging_setup).port,
                (logging_setup or click_logging_setup).log_level,
            ),
        )
    args_dict = args_dict | kwargs
    return args_dict


def get_process_pool_executor(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
) -> ProcessPoolExecutor:
    """Return a ProcessPool with some default settings."""
    args_dict = _pre_multiprocessing_args(nbr_cores, logging_setup, context, **kwargs)
    return ProcessPoolExecutor(**args_dict)


def get_pool_executor(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
) -> Pool:
    """Return a Pool with some default settings."""
    args_dict = _pre_multiprocessing_args(nbr_cores, logging_setup, context, **kwargs)
    nbr_of_processes = args_dict.pop("max_workers")
    args_dict.pop("mp_context")
    return multiprocessing.get_context(context).Pool(
        processes=nbr_of_processes, **args_dict
    )


R1_REGEX = R"(.[Rr]1$)|(_[Rr]?1$)|(_[Rr]?1)(?P<suffix>_[0-9]{3})$"
R2_REGEX = R"(.[Rr]2$)|(_[Rr]?2$)|(_[Rr]?2)(?P<suffix>_[0-9]{3})$"


def get_read_sample_name(read: str) -> str:
    """Extract the sample name from a read file.

    Strip fq.gz or fastq.gz extension and remove R1/R2 suffixes.
    Supported R1 R2 identifieds are:

    _R1,_R2 | _r1, _r2 | _1, _2 | .R1, .R2 | .r1, .r2

    :param read: filename of a fastq read file
    :return str: sample name
    :raise ValueError: if the read file does not have a valid extension
    """
    # group input file by sample id and order reads by R1 and R2
    if not (read.endswith("fq.gz") or read.endswith("fastq.gz")):
        raise ValueError("Invalid file extension: expected .fq.gz or .fastq.gz")

    read_stem = Path(read).name
    read_stem = read_stem.removesuffix(get_extension(read_stem, 2)).rstrip(".")
    r1_match = re.search(R1_REGEX, read_stem)
    r2_match = re.search(R2_REGEX, read_stem)

    # Check if the r1 and r2 suffixes are "exclusive or"
    if r1_match and r2_match or (not r1_match and not r2_match):
        raise ValueError("Invalid R1/R2 suffix.")

    # We need to cast away the optional here r1 or r2 will always
    # return a match object since we checked for both being None above
    match = typing.cast(re.Match[str], r1_match or r2_match)

    # Remove the R1 or R2 suffix by using the indices returned by the match
    s, e = match.span()
    sample_name = read_stem[0:s] + read_stem[e:-1]

    if match.groupdict().get("suffix"):
        sample_name += match.group("suffix")

    return sample_name


def is_read_file(read: Path | str, read_type: Literal["r1"] | Literal["r2"]) -> bool:
    """Check if a read filename matches the specified read_type.

    Detects the presence of a common read 1 or read 2 suffix in the filename.

    :param read: filename of a fastq read file
    :param read_type: the read type to check for (r1 or r2)
    :return bool: True if the read file is a read 1 or 2 file
    :raise ValueError: if the read file does not have a valid extension
    :raise AssertionError: if the read_type is not 'r1' or 'r2'
    """
    read = Path(read).name

    if read_type not in ("r1", "r2"):
        raise AssertionError("Invalid read type: expected 'r1' or 'r2'")

    if not (read.endswith("fq.gz") or read.endswith("fastq.gz")):
        raise ValueError("Invalid file extension: expected .fq.gz or .fastq.gz")

    match: re.Match[str] | None = None
    read_stem = Path(read.removesuffix(get_extension(read, 2)).rstrip(".")).name
    if read_type == "r1":
        match = re.search(R1_REGEX, read_stem)
    elif read_type == "r2":
        match = re.search(R2_REGEX, read_stem)
    else:
        raise AssertionError(
            "Invalid read type: could not find a read suffix in filename."
        )

    if not match:
        return False

    return True


def flatten(iterable: Iterable[Iterable[Any] | Any]) -> Generator[Any, None, None]:
    """Flatten an Iterable containing items or collection of items.

    Note: only list, set, tuple are flattened, strings and bytes are yielded as is

    :param iterable: list of lists or list of sets
    :return Generator[Any, None, None]: A generator yielding the flattened items
    :yield Any: the flattened items
    """
    for item in iterable:
        if isinstance(item, (str, bytes)):
            yield item
        elif isinstance(item, (list, set, tuple)):
            yield from item
        else:
            yield item
