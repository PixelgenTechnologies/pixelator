"""Pxl file abstraction for pixeldatasets.

A null pxl file is a valid pxl file that represents an empty sample produced
by a recoverable, data-caused failure. Its ``metadata`` JSON sets ``null`` to
true and stores a ``null_reason`` string. Downstream steps copy that file
instead of treating the empty sample as a software bug.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb

from pixelator.common.duckdb_utils import connect_duckdb
from pixelator.common.exceptions import PixelatorBaseException

PXL_FILE_MANDATOR_TABLES = [
    "__adata__X",
    "__adata__var",
    "__adata__obs",
    "edgelist",
]

# Should this be a "metadata" be a mandatory table?
PXL_FILE_ADATA_TABLES = ["__adata__X", "__adata__var", "__adata__obs", "__adata__uns"]
PXL_FILE_OTHER_TABLES = ["edgelist", "metadata", "layouts", "proximity"]


class NullPxlFileError(PixelatorBaseException):
    """Raised when a reader is asked to open a null pxl file.

    A null file means an upstream step produced no usable data for the sample.
    The stored reason is included in the message and on ``reason``.

    Attributes:
        path: Path of the null pxl file.
        reason: Why the upstream step produced no usable data.
    """

    def __init__(self, path: Path, reason: str) -> None:
        """Initialize the error.

        Args:
            path: Path of the null pxl file.
            reason: Why the file is null.
        """
        self.path = Path(path)
        self.reason = reason
        super().__init__(
            f"{self.path} is a null pxl file: an upstream step produced no usable "
            f"data for this sample. Reason: {reason}"
        )


class PxlFile:
    """PxlFile represents a a pxl file on disk and provides basic utility methods."""

    def __init__(self, path: Path, sample_name: str | None = None):
        """Initialize the PxlFile."""
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        self.path = path
        self._sample_name = sample_name

    @property
    def sample_name(self) -> str:
        """Return the sample name of the PxlFile."""
        if self._sample_name:
            return self._sample_name
        try:
            return self.metadata()["sample_name"]
        except KeyError:
            raise ValueError(
                f"Could not determine sample name from {self.path} - please provide a sample name."
            )

    def is_pxl_file(self) -> bool:
        """Check if the file is a PXL file."""
        with connect_duckdb(self.path, read_only=True) as con:
            tables = con.sql("SHOW ALL TABLES").to_df()
            return len(
                set(PXL_FILE_MANDATOR_TABLES).intersection(
                    set(tables["name"].to_list())
                )
            ) == len(PXL_FILE_MANDATOR_TABLES)

    def metadata(self) -> dict:
        """Read the metadata from the PXL file."""
        try:
            with connect_duckdb(self.path, read_only=True) as con:
                metadata = con.sql("SELECT * FROM metadata").fetchone()
                return json.loads(metadata[0]) if metadata else {}
        except duckdb.CatalogException:
            return {}

    def is_null_file(self) -> bool:
        """Return True when this file is a null pxl file."""
        return self.metadata().get("null") is True

    def null_reason(self) -> str | None:
        """Return why this file is null, or None when it is not null.

        Returns:
            The stripped reason string, or None.
        """
        if not self.is_null_file():
            return None
        reason = self.metadata().get("null_reason")
        if reason is None:
            return None
        text = str(reason).strip()
        return text or None

    def __repr__(self) -> str:
        """Return a string representation of the PxlFile."""
        return f"PxlFile({self.path})"

    def __str__(self) -> str:
        """Return a string representation of the PxlFile."""
        return f"{self.path}"

    @staticmethod
    def copy_pxl_file(src: PxlFile, target: Path) -> PxlFile:
        """Copy a PxlFile to a new location.

        Args:
            src: The source PxlFile.
            target: The target path.

        Returns:
            The new PxlFile.
        """
        shutil.copy(src.path, target)
        return PxlFile(target)


_EMPTY_NULL_TABLES_SQL = """
CREATE TABLE edgelist (
    umi1 UBIGINT,
    umi2 UBIGINT,
    read_count UINTEGER,
    marker_1 VARCHAR,
    marker_2 VARCHAR,
    component VARCHAR
);
CREATE TABLE "__adata__X" ("index" VARCHAR);
CREATE TABLE "__adata__var" ("index" VARCHAR);
CREATE TABLE "__adata__obs" ("index" VARCHAR);
CREATE TABLE "__adata__uns" (value JSON);
"""


def write_null_pxl(
    path: Path,
    *,
    sample_name: str,
    reason: str,
    panel_name: str | None = None,
    panel_version: str | None = None,
) -> PxlFile:
    """Write a null pxl file that records why a sample is empty.

    The file contains the mandatory pxl tables (empty) and metadata with
    ``null`` set and a non-empty ``null_reason``.

    Args:
        path: Destination ``.pxl`` path.
        sample_name: Sample the file stands in for.
        reason: Why the sample is null. Must be non-empty.
        panel_name: Antibody panel name, when known.
        panel_version: Antibody panel version, when known.

    Returns:
        The written null pxl file.

    Raises:
        ValueError: If ``reason`` is empty.
    """
    cleaned_reason = reason.strip()
    if not cleaned_reason:
        raise ValueError("A null pxl file requires a non-empty reason.")

    from pixelator import __version__
    from pixelator.pna.pixeldataset.io.pixel_file_writer import PixelFileWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "sample_name": sample_name,
        "version": __version__,
        "technology": "single-cell-pna",
        "null": True,
        "null_reason": cleaned_reason,
    }
    if panel_name is not None:
        metadata["panel_name"] = panel_name
    if panel_version is not None:
        metadata["panel_version"] = panel_version

    with PixelFileWriter(path, exits_ok=True) as writer:
        writer.get_connection().execute(_EMPTY_NULL_TABLES_SQL)
        writer.write_metadata(metadata)
    return PxlFile(path, sample_name=sample_name)


def reject_null_pxl(pxl_file: PxlFile) -> None:
    """Raise if ``pxl_file`` is a null pxl file.

    A null file with a reason raises :class:`NullPxlFileError`. A null file
    with no reason is a broken file and raises ``ValueError`` instead, so
    callers cannot treat it as a recoverable data failure.

    Args:
        pxl_file: Pxl file being opened.

    Raises:
        NullPxlFileError: If the file is null and stores a reason.
        ValueError: If the file is null but has no reason.
    """
    if not pxl_file.is_null_file():
        return
    reason = pxl_file.null_reason()
    if not reason:
        raise ValueError(
            f"{pxl_file.path} is a null pxl file but has no reason. "
            "This is not a recoverable data error."
        )
    raise NullPxlFileError(pxl_file.path, reason)
