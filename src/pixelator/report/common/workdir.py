"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
from __future__ import annotations

import enum
import functools
import itertools
import logging
import typing
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple

from pixelator.report.models import (
    CommandInfo,
    CommandOption,
)

from pixelator.types import PathType
from pixelator.utils import flatten, get_sample_name

logger = logging.getLogger(__name__)


class SingleCellStage(enum.Enum):
    AMPLICON = "amplicon"
    PREQC = "preqc"
    ADAPTERQC = "adapterqc"
    DEMUX = "demux"
    COLLAPSE = "collapse"
    GRAPH = "graph"
    ANNOTATE = "annotate"
    ANALYSIS = "analysis"
    REPORT = "report"


SingleCellStageLiteral = typing.Literal[
    "amplicon",
    "preqc",
    "adapterqc",
    "demux",
    "collapse",
    "graph",
    "annotate",
    "analysis",
    "report",
]

# Duplicating the keys is unavoidable here
WorkdirCacheKey: typing.TypeAlias = typing.Literal[
    "metadata",
    "single-cell amplicon",
    "single-cell preqc",
    "single-cell adapterqc",
    "single-cell demux",
    "single-cell collapse",
    "single-cell graph",
    "single-cell annotate",
    "single-cell annotate dataset",
    "single-cell annotate raw_components_metrics",
    "single-cell analysis",
    "single-cell report",
]

SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING: dict[
    SingleCellStageLiteral, WorkdirCacheKey
] = {
    "amplicon": "single-cell amplicon",
    "preqc": "single-cell preqc",
    "adapterqc": "single-cell adapterqc",
    "demux": "single-cell demux",
    "collapse": "single-cell collapse",
    "graph": "single-cell graph",
    "annotate": "single-cell annotate",
    "analysis": "single-cell analysis",
    "report": "single-cell report",
}


Model = typing.TypeVar("Model")


class PixelatorWorkdir:
    """Tools to manage and collect files from a pixelator workdir folder."""

    #: A dict that maps a key (usually a key per subcommand) to a glob pattern
    #: that will be used to search for report files in the workdir folder.
    _SEARCH_PATTERNS: Dict[WorkdirCacheKey, str] = {
        "metadata": "**/*.meta.json",
        "single-cell amplicon": "amplicon/*.report.json",
        "single-cell preqc": "preqc/*.report.json",
        "single-cell adapterqc": "adapterqc/*.report.json",
        "single-cell demux": "demux/*.report.json",
        "single-cell collapse": "collapse/*.report.json",
        "single-cell graph": "graph/*.report.json",
        "single-cell annotate": "annotate/*.report.json",
        "single-cell annotate dataset": "annotate/*.dataset.pxl",
        "single-cell annotate raw_components_metrics": "annotate/*.raw_components_metrics.csv.gz",
        "single-cell analysis": "analysis/*.report.json",
        "single-cell report": "report/*.report.json",
    }

    _SEARCH_ANNOTATE_DATASET = "annotate/*.dataset.pxl"

    def __init__(self, basedir: PathType):
        self.basedir = Path(basedir)
        self._cache: dict[WorkdirCacheKey, dict[str, Path]] = {}
        self._sample_ids: set[str] | None = None

    def scan(self):
        """Retrieve all reports and metadata from the workdir folder."""
        self.metadata_files()
        for stage in SingleCellStage:
            # Some stages may not be present in the workdir, but
            # we still collect whatever is found
            try:
                self.single_cell_report(stage, cache=False)
            except FileNotFoundError:
                continue

    def stage_dir(self, stage: SingleCellStage | SingleCellStageLiteral):
        """Create a subcommand directory.

        :param stage: The subcommand to create a directory for
        :return: The absolute path to the subcommand directory
        """
        dirname = stage.value if isinstance(stage, enum.Enum) else stage
        subcommand_path = self.basedir / dirname
        subcommand_path.mkdir(exist_ok=True, parents=True)
        return subcommand_path

    def _check_folder(self, name: str) -> Path:
        """Check that the directory for a specific subcommand is present.

        :param name: The name of the subcommand folder
        :returns: The absolute path to the subcommand folder
        :raises FileNotFoundError: If the folder is missing
        """
        source_path = Path(self.basedir) / name
        if not source_path.is_dir():
            raise FileNotFoundError(f"{name} folder missing in {self.basedir}")

        return source_path.absolute()

    def samples(self, cache: bool = True) -> set[str]:
        """Return a list of all sample_ids encountered in the current workdir."""
        if cache and self._sample_ids is not None:
            return self._sample_ids

        self.scan()
        sids: set[str] = set()
        for pattern, data in self._cache.items():
            sids.update(data.keys())

        self._sample_ids = sids
        return self._sample_ids

    def _check_pattern_basedir(self, pattern_id: WorkdirCacheKey) -> Path:
        pattern = PurePath(self._SEARCH_PATTERNS[pattern_id])
        pattern_dir_levels = len(pattern.parts)

        basedir: PurePath
        if not pattern.is_absolute():
            basedir = pattern.parents[-pattern_dir_levels]
        else:
            rel_pattern = pattern.relative_to(self.basedir)
            basedir = rel_pattern.parents[-pattern_dir_levels]

        return self._check_folder(str(basedir))

    def _collect_files_by_pattern(
        self, pattern: str, prefix: Optional[Path] = None
    ) -> List[Tuple[str, List[Path]]]:
        """Collect a set of report files from a subcommand folder.

        :param pattern: A glob pattern to search for.
        :param prefix: A prefix path to search in relative to the workdir
        """
        # collect the metrics files
        search_dir = self.basedir / prefix if prefix else self.basedir

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

    def _collect_files_by_key(
        self, search_key: WorkdirCacheKey, prefix: Optional[Path] = None
    ) -> List[Tuple[str, List[Path]]]:
        """Collect a set of report files from a subcommand folder.

        :param search_key: The key of an entry in the _SEARCH_PATTERNS dict
        :param prefix: A prefix path to search in relative to the workdir
        """
        # collect the metrics files
        search_dir = self.basedir / prefix if prefix else self.basedir

        pattern = self._SEARCH_PATTERNS[search_key]
        return self._collect_files_by_pattern(pattern, prefix)

    def _collect_metadata_files(self) -> dict[str, list[Path]]:
        """
        Collect all metadata files.

        Metadata files are those that end with `.meta.json`.
        These typically contains information about the pixelator version
        and the commandline line arguments used to run a specific subcommand.
        The files will be returned as a flat list

        :raises AssertionError: If no metadata files are found
        :return: A list of metadata files
        """

        files = self._collect_files_by_key("metadata")
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

    def _collect_output_files(self, pattern_id) -> dict[str, Path]:
        """Collect output files from a pixelator workdir.

        The pattern id is used to fetch a glob pattern from py::attr:`_SEARCH_PATTERNS`
        this glob is relative to the workdir folder.

        :param pattern_id: The pattern id to use for searching
        """
        self._check_pattern_basedir(pattern_id)
        collected_files = self._collect_files_by_key(pattern_id)
        stage_reports_index = {}

        for sample, filelist in collected_files:
            if len(filelist) != 1:
                raise AssertionError(
                    f"Expected exactly one report file for sample {sample}"
                )
            stage_reports_index[sample] = filelist[0]

        return stage_reports_index

    def metadata_files(
        self, sample: Optional[str] = None, cache: bool = True
    ) -> list[Path]:
        """Return a list of metadata files.

        :param sample: The sample name to filter on
        :param cache: Whether to return cached result if available
        :return: A list of metadata files
        """
        cache_key: WorkdirCacheKey = "metadata"

        def func():
            report_per_sample = self._collect_metadata_files()

            out: typing.MutableMapping[str, list[Path]] = defaultdict(list)
            for k, v in report_per_sample.items():
                if isinstance(v, list):
                    out[k].extend(v)
                else:
                    out[k].append(v)

            return out

        return self._cached_reports_implementation(cache_key, func, sample, cache)

    def _cached_reports_implementation(
        self,
        cache_key: WorkdirCacheKey,
        fn: typing.Callable[[], dict[str, list[Path]]],
        sample: Optional[str] = None,
        cache: bool = True,
    ) -> Path | List[Path]:
        if not cache or self._cache.get(cache_key) is None:
            self._cache[cache_key] = fn()

        if sample:
            return self._cache[cache_key][sample]

        return list(flatten(self._cache[cache_key].values()))

    @typing.overload
    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: None = None,
        *,
        cache: bool = True,
    ) -> List[Path]:
        ...

    @typing.overload
    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: str,
        *,
        cache: bool = True,
    ) -> Path:
        ...

    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: Optional[str] = None,
        *,
        cache: bool = True,
    ) -> Path | List[Path]:
        """Retrieve a report from a stage of the single-cell pipeline.

        :param stage_name: The name of the stage to retrieve a report from
        :param sample: The sample name to filter on
        :param cache: Whether to return cached result if available
        :return: A list of metadata files
        """
        stage_key = stage.value if isinstance(stage, enum.Enum) else stage
        cache_key: WorkdirCacheKey = SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING[stage_key]
        if cache_key is None:
            return []

        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)

    @typing.overload
    def filtered_dataset(
        self,
        sample: None = None,
        cache: bool = True,
    ) -> list[Path]:
        ...

    @typing.overload
    def filtered_dataset(
        self,
        sample: str,
        cache: bool = True,
    ) -> Path:
        ...

    def filtered_dataset(self, sample: str, cache: bool = True) -> Path | list[Path]:
        """Return the path to a filtered dataset for a sample.

        This is the output `dataset.pxl` file from single-cell annotate

        :param sample: The sample name to retrieve the filtered dataset for
        """
        cache_key: WorkdirCacheKey = "single-cell annotate dataset"
        if cache_key is None:
            return []

        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)

    @typing.overload
    def raw_component_metrics(
        self,
        sample: None = None,
        cache: bool = True,
    ) -> list[Path]:
        ...

    @typing.overload
    def raw_component_metrics(
        self,
        sample: str,
        cache: bool = True,
    ) -> Path:
        ...

    def raw_component_metrics(
        self, sample: Optional[str] = None, cache: bool = True
    ) -> Path | list[Path]:
        """Return the path to a raw_component_metrics file.

        This is generated by single-cell annotate

        :param sample: The sample name to retrieve the filtered dataset for
        """
        cache_key: WorkdirCacheKey = "single-cell annotate raw_components_metrics"
        if cache_key is None:
            return []

        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)
