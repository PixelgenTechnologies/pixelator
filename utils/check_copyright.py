#!/usr/bin/env python3
"""Copyright (c) 2023 Pixelgen Technologies AB.

This is a general copyright checker for python files committed to pixelator

Do not delete the shebang on top of the file or it will stop working
"""
import ast
import sys
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

ROOT_DIR = Path(__file__).parent / ".."
TEST_DIR = ROOT_DIR / "tests"
RND_DIR = ROOT_DIR / "src/pixelator_rnd"
RND_CONFIG_DIR = ROOT_DIR / "src/pixelator_rnd_config"
TISSUE_DIR = ROOT_DIR / "src/pixelator_tissue"
PYTHON_DIRS = [RND_DIR, TEST_DIR]


class CopyrightNoticeMissing(Exception):
    """Copyright notice class.

    This is a class extension of the :class:`Exception` class in order to
    collect missing copyright in python source files.

    :param message: a message of exception
    :param offending_file: the file with no copyright
    """

    def __init__(self, message: str, offending_file: Path) -> None:
        """Construct instance."""
        super().__init__(message)
        self.file = offending_file


def check_file_for_copyright(py_file: Path) -> Optional[CopyrightNoticeMissing]:
    """Check file for presence of copyright notice.

    :param py_file: a python file
    :return: missing notice class if copyright not present in file
    :rtype: Optional[CopyrightNoticeMissing]
    """
    raw_tree = py_file.read_text()
    tree = ast.parse(raw_tree)
    module_docstring = ast.get_docstring(tree, clean=True)
    if not module_docstring:
        return CopyrightNoticeMissing("Module docstring missing", py_file.resolve())
    if "Copyright (c)" not in module_docstring:
        return CopyrightNoticeMissing(
            "Copyright notice missing from module docstring", py_file.resolve()
        )
    return None


def check_copyright(files: Optional[Iterator[Path]]):
    """Check a list of files for copyright.

    :param files: files to check
    """

    def errors():
        files_to_check = files or chain.from_iterable(
            [directory.rglob("**/*.py") for directory in PYTHON_DIRS]
        )
        for py_file in files_to_check:
            error = check_file_for_copyright(py_file)
            if error:
                yield error

    found_errors = list(errors())
    if found_errors:
        print("A copyright notice is missing from the following files:")
        for exception in found_errors:
            print(exception.file, str(exception), sep=": ")
        sys.exit(1)

    print("All .py files have a copyright notice")


# Add arguments to script to check list of files
if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_copyright(map(Path, sys.argv[1:]))
    else:
        check_copyright(None)
