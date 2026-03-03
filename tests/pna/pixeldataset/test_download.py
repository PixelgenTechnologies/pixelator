"""Tests for download module.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import pixelator.pna.pixeldataset.download as dl


class FakeResponse:
    def __init__(self, *, chunks: list[bytes], headers: dict[str, str] | None = None):
        self._chunks = chunks
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):  # noqa: ARG002
        yield from self._chunks


def test_download_dataset_raises_for_unknown_dataset():
    with pytest.raises(ValueError, match=r"Dataset does-not-exist not found"):
        dl.DownloadableDatasets.download_dataset("does-not-exist")


def test_download_dataset_picks_latest_version_by_default_and_default_output_path(
    monkeypatch, run_in_tmpdir
):
    dataset_v1 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=1,
        url="https://example.com/v1",
    )
    dataset_v2 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=2,
        url="https://example.com/v2",
    )
    monkeypatch.setattr(
        dl,
        "_DATASET_MAPPINGS",
        {"example": {1: dataset_v1, 2: dataset_v2}},
    )

    called: dict[str, object] = {}

    def fake_download(url: str, output_path: Path) -> Path:
        called["url"] = url
        called["output_path"] = output_path
        return output_path

    monkeypatch.setattr(dl, "_download_pixel_dataset", fake_download)

    out = dl.DownloadableDatasets.download_dataset("example")
    assert out == Path("./pixelator-datasets/example.layout.pxl")
    assert called["url"] == "https://example.com/v2"
    assert called["output_path"] == Path("./pixelator-datasets/example.layout.pxl")


def test_download_dataset_respects_explicit_version(monkeypatch):
    dataset_v1 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=1,
        url="https://example.com/v1",
    )
    dataset_v2 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=2,
        url="https://example.com/v2",
    )
    monkeypatch.setattr(
        dl,
        "_DATASET_MAPPINGS",
        {"example": {1: dataset_v1, 2: dataset_v2}},
    )

    called: dict[str, object] = {}

    def fake_download(url: str, output_path: Path) -> Path:
        called["url"] = url
        called["output_path"] = output_path
        return output_path

    monkeypatch.setattr(dl, "_download_pixel_dataset", fake_download)

    out_path = Path("somewhere.layout.pxl")
    out = dl.DownloadableDatasets.download_dataset(
        "example", version=1, output_path=out_path
    )
    assert out == out_path
    assert called["url"] == "https://example.com/v1"
    assert called["output_path"] == out_path


def test_download_dataset_skips_if_file_exists_and_overwrite_is_false(
    monkeypatch, tmp_path
):
    dataset_v1 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=1,
        url="https://example.com/v1",
    )
    monkeypatch.setattr(dl, "_DATASET_MAPPINGS", {"example": {1: dataset_v1}})

    output_path = tmp_path / "example.layout.pxl"
    output_path.write_bytes(b"already-there")

    monkeypatch.setattr(
        dl,
        "_download_pixel_dataset",
        lambda *_args, **_kwargs: pytest.fail("_download_pixel_dataset should not run"),
    )

    out = dl.DownloadableDatasets.download_dataset(
        "example",
        version=1,
        output_path=output_path,
        overwrite=False,
    )
    assert out == output_path


def test_download_dataset_overwrites_if_overwrite_true(monkeypatch, tmp_path):
    dataset_v1 = dl.Dataset(
        name="example",
        description="Example dataset",
        version=1,
        url="https://example.com/v1",
    )
    monkeypatch.setattr(dl, "_DATASET_MAPPINGS", {"example": {1: dataset_v1}})

    output_path = tmp_path / "example.layout.pxl"
    output_path.write_bytes(b"already-there")

    called: dict[str, object] = {}

    def fake_download(url: str, output_path: Path) -> Path:
        called["url"] = url
        called["output_path"] = output_path
        return output_path

    monkeypatch.setattr(dl, "_download_pixel_dataset", fake_download)

    out = dl.DownloadableDatasets.download_dataset(
        "example",
        version=1,
        output_path=output_path,
        overwrite=True,
    )
    assert out == output_path
    assert called["url"] == "https://example.com/v1"
    assert called["output_path"] == output_path


def test_download_pixel_dataset_writes_file_and_calls_requests_get(
    monkeypatch, tmp_path
):
    called: dict[str, object] = {}

    def fake_get(url: str, *, stream: bool, timeout: tuple[int, int]):
        called["url"] = url
        called["stream"] = stream
        called["timeout"] = timeout
        return FakeResponse(chunks=[b"abc", b"def"], headers={"content-length": "6"})

    monkeypatch.setattr(dl.requests, "get", fake_get)

    progress: list[str] = []

    def fake_report(msg: str, *args: object) -> None:
        progress.append(msg % args if args else msg)

    monkeypatch.setattr(dl, "_report_progress", fake_report)

    output_path = tmp_path / "nested" / "file.layout.pxl"
    out = dl._download_pixel_dataset("https://example.com/file", output_path)

    assert out == output_path
    assert output_path.read_bytes() == b"abcdef"
    assert called["url"] == "https://example.com/file"
    assert called["stream"] is True
    assert called["timeout"] == (dl._CONNECT_TIMEOUT, dl._READ_TIMEOUT)
    assert any("Starting download from" in p for p in progress)
    assert any("Download progress:" in p for p in progress)
    assert any("Download complete:" in p for p in progress)


def test_report_progress_prints_in_interactive_mode(monkeypatch, capsys):
    monkeypatch.setattr(dl, "_is_interactive", lambda: True)
    dl._report_progress("Hello %s", "world")
    out = capsys.readouterr().out
    assert "Hello world" in out


def test_report_progress_logs_in_non_interactive_mode(monkeypatch, mocker):
    monkeypatch.setattr(dl, "_is_interactive", lambda: False)
    info = mocker.Mock()
    monkeypatch.setattr(dl.logger, "info", info)

    dl._report_progress("Hello %s", "world")
    info.assert_called_once_with("Hello world")


def test_is_interactive_detects_ipython(monkeypatch):
    monkeypatch.setattr(dl.sys.stdout, "isatty", lambda: False)
    monkeypatch.setitem(
        dl.sys.modules,
        "IPython",
        types.SimpleNamespace(get_ipython=lambda: object()),
    )
    assert dl._is_interactive() is True
