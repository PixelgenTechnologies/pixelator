"""Tests for null pxl files."""

import pytest

from pixelator.pna.pixeldataset.io import (
    PxlFile,
    null_passthrough_reason,
    write_null_pxl,
)
from pixelator.pna.pixeldataset.io.pixel_file_writer import PixelFileWriter


def test_write_null_pxl_stores_reason_and_is_a_pxl_file(tmp_path):
    """A null file is a valid pxl file and round-trips its reason."""
    path = tmp_path / "empty.sample.pxl"
    written = write_null_pxl(
        path,
        sample_name="empty.sample",
        reason="  No cells above the component size threshold.  ",
        panel_name="proxiome-v1",
        panel_version="1.0",
    )

    assert written.is_pxl_file()
    assert written.is_null()
    assert written.null_reason() == "No cells above the component size threshold."
    assert written.sample_name == "empty.sample"
    metadata = written.metadata()
    assert metadata["null"] is True
    assert metadata["panel_name"] == "proxiome-v1"

    copied = PxlFile.copy_pxl_file(written, tmp_path / "copied.pxl")
    assert copied.is_null()
    assert null_passthrough_reason(copied) == written.null_reason()


def test_write_null_pxl_rejects_empty_reason(tmp_path):
    """A null file without a reason is not a valid recoverable failure."""
    with pytest.raises(ValueError, match="non-empty reason"):
        write_null_pxl(tmp_path / "bad.pxl", sample_name="bad", reason="   ")


def test_regular_pxl_is_not_null(tmp_path):
    """Metadata without the null flag is a normal file."""
    path = tmp_path / "ok.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_metadata({"sample_name": "ok", "null": False})
    pxl_file = PxlFile(path)
    assert pxl_file.is_null() is False
    assert pxl_file.null_reason() is None
    assert null_passthrough_reason(pxl_file) is None


def test_null_pxl_without_reason_is_unrecoverable(tmp_path):
    """A null flag with no reason must not be passed through."""
    path = tmp_path / "broken.pxl"
    with PixelFileWriter(path) as writer:
        writer.write_metadata({"sample_name": "broken", "null": True})
    pxl_file = PxlFile(path)
    assert pxl_file.is_null() is True
    with pytest.raises(ValueError, match="no reason"):
        null_passthrough_reason(pxl_file)
