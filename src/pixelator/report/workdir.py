"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pixelator.types import PathType
from pixelator.utils import get_sample_name


class PixelatorWorkdir:
    """
    Tools to collect files from the workdir folder.
    """

    _SEARCH_PATTERNS = {
        "metadata": "**/*.meta.json",
    }

    def __init__(self, basedir: PathType):
        self.basedir = Path(basedir)
        self._metadata = self._collect_metadata_files()

    def _check_folder(self, name: str) -> None:
        """Check that the directory for a specific subcommand is present."""
        source_path = Path(self.basedir) / name
        if not source_path.is_dir():
            raise AssertionError(f"{name} folder missing in {self.basedir}")

    def _collect_files(
        self, file_type: str, subcommand_path: Optional[Path] = None
    ) -> List[Tuple[str, List[Path]]]:
        # collect the metrics files
        search_dir = self.basedir / subcommand_path if subcommand_path else self.basedir

        pattern = self._SEARCH_PATTERNS[file_type]

        files = list(search_dir.rglob(pattern))

        tuples = []
        for f in files:
            sample_name = get_sample_name(f)
            tuples.append((sample_name, f))

        tuples.sort(key=lambda x: x[0])
        grouped_files = [
            (str(sample), [p for (s, p) in grouper])
            for sample, grouper in itertools.groupby(tuples, key=lambda x: x[0])
        ]
        return grouped_files

    def _collect_metadata_files(self) -> Dict[str, List[Path]]:
        """
        Collect all metadata files.

        Metadata files are those that end with `.meta.json`.
        The files will be returned as a flat list

        :raises AssertionError: If no metadata files are found
        :return: A list of metadata files
        """

        files = self._collect_files("metadata")
        if files is None or len(files) == 0:
            logging.warning(f"No metadata files found in {self.basedir}")
            return {}

        flat_files = list(
            itertools.chain.from_iterable(paths for sample, paths in files)
        )

        metadata_files_index = defaultdict(list)

        for file in flat_files:
            metadata_files_index[get_sample_name(file.name)].append(file)

        return {**metadata_files_index}

    def metadata_files(self, sample: Optional[str] = None) -> List[Path]:
        if sample:
            return self._metadata[sample]

        return list(itertools.chain.from_iterable(self._metadata.values()))
