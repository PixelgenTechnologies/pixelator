"""
This module contains all the extra exception classes and handling
defined by pixelator

Copyright (c) 2022 Pixelgen Technologies AB.
"""

from pathlib import Path
from typing import Union


class FileFqGzEmpty(Exception):
    """
    Class to manage empty fastq.gz file exceptions.

    Attributes:
        msg: the error message to output
        fname: the name of the file
        size: the size of the file uncompressed (should be 0)
    """

    def __init__(self, msg: str, fname: Union[str, Path], size: int):
        self.msg = msg
        self.fname = fname
        self.size = size
