"""Module for downloading pixel datasets that can be used with e.g. tutorials.

Copyright © 2026 Pixelgen Technologies AB.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Metadata for a downloadable dataset."""

    name: str
    description: str
    version: int
    url: str


# List of available datasets
_DATASETS = [
    Dataset(
        name="pna062-pha-pbmcs",
        url="https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/pna-datasets/v2/sha-079762b-2025-07-02-74062c09/pixelator/PNA062_PHA_PBMCs_1000cells_S04_S4.layout.pxl?download=1",
        description="PHA stimulated ~1000 cells PBMCs",
        version=1,
    ),
    Dataset(
        name="pna062-unstim-pbmcs",
        url="https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/pna-datasets/v2/sha-079762b-2025-07-02-74062c09/pixelator/PNA062_unstim_PBMCs_1000cells_S02_S2.layout.pxl?download=1",
        description="Unstimulated ~1000 cells PBMCs",
        version=1,
    ),
]

# The structure here is name -> {version -> Dataset}
_DATASET_MAPPINGS = {dataset.name: {dataset.version: dataset} for dataset in _DATASETS}


class DownloadableDatasets:
    """Download example pixel datasets for tutorials and testing.

    Use this class to fetch small example datasets (e.g. PNA PBMCs) to your
    machine. Call ``list_datasets()`` to see what is available, then
    ``download_dataset()`` with the dataset name to download it. If the file
    already exists at the destination, the download is skipped unless you pass
    ``overwrite=True``.

    Example:
        List available datasets and download one::

            from pathlib import Path
            from pixelator.pna.pixeldataset.download import DownloadableDatasets

            # See what datasets exist
            DownloadableDatasets.list_datasets()

            # Download the latest version to the default folder (./pixelator-datasets/{dataset_name}.layout.pxl)
            # i.e. in this case ./pixelator-datasets/pna062-pha-pbmcs.layout.pxl
            path = DownloadableDatasets.download_dataset("pna062-pha-pbmcs")

            # Download to a specific path
            path = DownloadableDatasets.download_dataset(
                "pna062-unstim-pbmcs",
                output_path=Path("./data/my_dataset.layout.pxl"),
            )

            # Re-download and overwrite an existing file
            path = DownloadableDatasets.download_dataset(
                "pna062-pha-pbmcs",
                output_path=Path("./data/example.layout.pxl"),
                overwrite=True,
            )

    """

    @staticmethod
    def download_dataset(
        dataset_name: str,
        version: int | None = None,
        output_path: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Download a dataset from the URL to a local path.

        If `output_path` is not provided, the dataset will be downloaded into a subdirectory called
        `pixelator-datasets` in the current working directory.

        Args:
        dataset_name: The name of the dataset to download.
        version: The version of the dataset to download. If not provided, the latest version will be downloaded.
        output_path: The path to save the dataset to. Defaults to `./pixelator-datasets/{dataset_name}.layout.pxl`
        overwrite: If False and a file already exists at the destination, do not download and return the path. If True, download again and overwrite the existing file.

        Raises:
        ValueError: If the dataset is not found.

        """
        if dataset_name not in _DATASET_MAPPINGS:
            raise ValueError(
                f"Dataset {dataset_name} not found. Use `DownloadableDatasets.list_datasets()` to see all available datasets."
            )
        if version is None:
            version = max(
                dataset.version for dataset in _DATASET_MAPPINGS[dataset_name].values()
            )

        if output_path is None:
            output_path = Path(f"./pixelator-datasets/{dataset_name}.layout.pxl")

        if output_path.exists() and output_path.is_file() and not overwrite:
            _report_progress(
                "File already exists at %s. Use overwrite=True to download again.",
                output_path,
            )
            return output_path
        url = _DATASET_MAPPINGS[dataset_name][version].url
        return _download_pixel_dataset(url, output_path)

    @staticmethod
    def list_datasets():
        """List all available datasets."""
        print("Available datasets:")
        for dataset in _DATASETS:
            print(f"- {dataset}")

    def __str__(self) -> str:
        """Return the string representation of the class."""
        return f"DownloadableDatasets(datasets={list(_DATASET_MAPPINGS.keys())})"

    def __repr__(self) -> str:
        """Return the string representation of the class."""
        return str(self)

    def _ipython_display_(self):
        """Display the DownloadableDatasets in Jupyter notebooks."""
        print(self.list_datasets())


# Chunk size for streaming download (8 MB - good balance for large files)
_DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024

# Connection timeout in seconds (how long to wait for server response)
_CONNECT_TIMEOUT = 30

# Read timeout in seconds (per chunk - allows long total download time)
_READ_TIMEOUT = 300


def _is_interactive() -> bool:
    """Return True if running in an interactive environment (TTY, Jupyter, IPython)."""
    if sys.stdout.isatty():
        return True
    try:
        get_ipython = sys.modules["IPython"].get_ipython  # noqa: S105
        if get_ipython() is not None:
            return True
    except (KeyError, AttributeError):
        pass
    return False


def _report_progress(msg: str, *args: object) -> None:
    """Report progress to stdout in interactive environments, otherwise to logger.

    Args:
    msg: Msg.
    args: Args.

    """
    formatted = msg % args if args else msg
    if _is_interactive():
        print(formatted, flush=True)
    else:
        logger.info(formatted)


def _download_pixel_dataset(url: str, output_path: Path) -> Path:
    """Download a pixel dataset from a URL to a local path.

    Uses streaming to handle large files (GB range) without loading them
    into memory. Progress is logged periodically.

    Args:
    url: The URL of the pixel dataset to download.
    output_path: The path to save the pixel dataset to.

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _report_progress("Starting download from %s to %s", url, output_path)

    with requests.get(
        url,
        stream=True,
        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
    ) as response:
        response.raise_for_status()

        total_size = response.headers.get("content-length")
        total_bytes = int(total_size) if total_size else None

        bytes_downloaded = 0
        last_logged_pct = -1

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bytes_downloaded += len(chunk)

                    if total_bytes and total_bytes > 0:
                        pct = int(100 * bytes_downloaded / total_bytes)
                        # Log progress every 5%
                        if pct >= last_logged_pct + 5:
                            _report_progress(
                                "Download progress: %d%% (%d / %d MB)",
                                pct,
                                bytes_downloaded // (1024 * 1024),
                                total_bytes // (1024 * 1024),
                            )
                            last_logged_pct = pct
                    else:
                        mb = bytes_downloaded // (1024 * 1024)
                        if mb > 0 and mb % 100 == 0 and mb != last_logged_pct:
                            _report_progress("Download progress: %d MB downloaded", mb)
                            last_logged_pct = mb

    _report_progress(
        "Download complete: %s (%d MB)",
        output_path,
        bytes_downloaded // (1024 * 1024),
    )
    return output_path
