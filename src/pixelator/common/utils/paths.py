"""Utility functions for working with file paths and validating input files.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import collections.abc
import gzip
import logging
from pathlib import Path, PurePath
from typing import Optional, Sequence, Union

from pixelator.common.types import PathType

logger = logging.getLogger(__name__)


def create_output_stage_dir(root: PathType, name: str) -> Path:
    """Create a new subfolder with `name` under the given `root` directory.

    Args:
        root: the parent directory
        name: the name of the directory to create

    Returns:
        the created folder (Path) (Path)
    """
    output = Path(root) / name
    if not output.is_dir():
        output.mkdir(parents=True)
    return output


def get_extension(filename: PathType, len_ext: int = 2) -> str:
    """Extract file extensions from a filename.

    Args:
        filename: the file name
        len_ext: the number of expected extensions parts e.g.: fq.gz gives len_ext=2

    Returns:
        the file extension (str) (str)
    """
    return "".join(PurePath(filename).suffixes[-len_ext:]).lstrip(".")


def get_sample_name(filename: PathType) -> str:
    """Extract the sample name from a sample's filename.

    The sample name is expected to be from the start of the filename until
    the first dot.

    Args:
        filename: path to the file

    Returns:
        the sample name (str)
    """
    return Path(filename).stem.split(".")[0]


def gz_size(filename: str) -> int:
    """Extract the size of a gzip compressed file.

    Args:
        filename: file name

    Returns:
        size of the file uncompressed (in bits) (int)
    """
    with gzip.open(filename, "rb") as f:
        return f.seek(0, whence=2)


def sanity_check_inputs(
    input_files: Sequence[PathType] | PathType,
    allowed_extensions: Union[Sequence[str], Optional[str]] = None,
) -> None:
    """Perform basic sanity checking of input files.

    Args:
        input_files: the files to sanity check
        allowed_extensions: the expected file extension of the files, e.g. 'fastq.gz' or a tuple of
            allowed types eg. ('fastq.gz', 'fq.gz')

    Returns:
        None

    Raises:
        AssertionError: when any of validation fails
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
