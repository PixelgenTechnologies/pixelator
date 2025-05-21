"""Copyright © 2023 Pixelgen Technologies AB."""

from __future__ import annotations

import enum
import functools
import itertools
import logging
import typing
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple

from pixelator.common.types import PathType
from pixelator.common.utils import flatten, get_sample_name

logger = logging.getLogger(__name__)


class SingleCellStage(enum.Enum):
    """Enum for the different stages of the single-cell pipeline."""

    AMPLICON = "amplicon"
    PREQC = "preqc"
    ADAPTERQC = "adapterqc"
    DEMUX = "demux"
    COLLAPSE = "collapse"
    GRAPH = "graph"
    ANNOTATE = "annotate"
    LAYOUT = "layout"
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
    "layout",
    "analysis",
    "report",
]

# Duplicating the keys is unavoidable here
WorkdirCacheKey: typing.TypeAlias = typing.Literal[
    "metadata",
    "single-cell-mpx amplicon",
    "single-cell-mpx preqc",
    "single-cell-mpx adapterqc",
    "single-cell-mpx demux",
    "single-cell-mpx collapse",
    "single-cell-mpx graph",
    "single-cell-mpx annotate",
    "single-cell-mpx annotate dataset",
    "single-cell-mpx annotate raw_components_metrics",
    "single-cell-mpx layout",
    "single-cell-mpx analysis",
    "single-cell-mpx report",
]

SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING: dict[
    SingleCellStageLiteral, WorkdirCacheKey
] = {
    "amplicon": "single-cell-mpx amplicon",
    "preqc": "single-cell-mpx preqc",
    "adapterqc": "single-cell-mpx adapterqc",
    "demux": "single-cell-mpx demux",
    "collapse": "single-cell-mpx collapse",
    "graph": "single-cell-mpx graph",
    "annotate": "single-cell-mpx annotate",
    "layout": "single-cell-mpx layout",
    "analysis": "single-cell-mpx analysis",
    "report": "single-cell-mpx report",
}


CacheDictType = dict[WorkdirCacheKey, dict[str, Path] | dict[str, list[Path]]]


class WorkdirOutputNotFound(Exception):
    """Raised when an output file is is not found for a specific sample."""

    def __init__(
        self,
        *args,
        message: str | None = None,
        sample_id: str | None = None,
        pattern: str | None = None,
        workdir: Path | None = None,
    ) -> None:
        """Initialize the exception.

        :param args: Positional arguments to pass to the base
        :param message: A custom message to use
        :param sample_id: The sample id for which the output was not found
        :param pattern: The pattern used to search for the output
        :param workdir: The workdir folder where the output was searched for
        """
        super().__init__(*args)
        self.sample_id = sample_id
        self.pattern = pattern
        self.workdir = workdir
        self.message = message or (
            f'No output files found for sample "{self.sample_id}":\n'
            f'  using pattern "{self.pattern}" in "{self.workdir}"'
        )

    def __str__(self):
        """Return a string representation of the exception."""
        return self.message


class PixelatorWorkdir:
    """Tools to manage and collect files from a pixelator workdir folder."""

    # A dict that maps a key (usually a key per subcommand) to a glob pattern
    # that will be used to search for report files in the workdir folder.
    _SEARCH_PATTERNS: Dict[WorkdirCacheKey, str] = {
        "metadata": "**/*.meta.json",
        "single-cell-mpx amplicon": "amplicon/*.report.json",
        "single-cell-mpx preqc": "preqc/*.report.json",
        "single-cell-mpx adapterqc": "adapterqc/*.report.json",
        "single-cell-mpx demux": "demux/*.report.json",
        "single-cell-mpx collapse": "collapse/*.report.json",
        "single-cell-mpx graph": "graph/*.report.json",
        "single-cell-mpx annotate": "annotate/*.report.json",
        "single-cell-mpx annotate dataset": "annotate/*.dataset.pxl",
        "single-cell-mpx annotate raw_components_metrics": "annotate/*.raw_components_metrics.csv.gz",
        "single-cell-mpx layout": "layout/*.report.json",
        "single-cell-mpx analysis": "analysis/*.report.json",
        "single-cell-mpx report": "report/*.report.json",
    }

    _SEARCH_ANNOTATE_DATASET = "annotate/*.dataset.pxl"

    def __init__(self, basedir: PathType):
        """Initialize the PixelatorWorkdir object."""
        self.basedir = Path(basedir)
        self._cache: CacheDictType = {}
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
        basedir = pattern.parents[0]
        return self._check_folder(str(basedir))

    def _collect_files_by_pattern(
        self, pattern: str, prefix: Optional[Path] = None
    ) -> List[Tuple[str, List[Path]]]:
        """Collect a set of report files from a subcommand folder.

        :param pattern: A glob pattern to search for.
        :param prefix: A prefix path to search in relative to the workdir
        :return: A list of tuples with sample name and a list of report files
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

        :param search_key: The key of an entry in the `ivar:_SEARCH_PATTERNS` dict
        :param prefix: A prefix path to search in relative to the workdir
        :return: A list of tuples with sample name and a list of report files
        """
        # collect the metrics files
        pattern = self._SEARCH_PATTERNS[search_key]
        return self._collect_files_by_pattern(pattern, prefix)

    def _collect_metadata_files(self) -> dict[str, list[Path]]:
        """Collect all metadata files.

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

    def _collect_output_files(self, pattern_id: WorkdirCacheKey) -> dict[str, Path]:
        """Collect output files from a pixelator workdir.

        The pattern id is used to fetch a glob pattern from py::attr:`_SEARCH_PATTERNS`
        this glob is relative to the workdir folder.

        :param pattern_id: The pattern id to use for searching
        :return: A dictionary with sample names as keys and the output file as value
        :raises AssertionError: If there is more than 1 match with the pattern
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

        def func() -> dict[str, list[Path]]:
            report_per_sample = self._collect_metadata_files()

            out: dict[str, list[Path]] = defaultdict(list)
            for k, v in report_per_sample.items():
                if isinstance(v, list):
                    out[k].extend(v)
                else:
                    out[k].append(v)

            return out

        return self._cached_reports_implementation(cache_key, func, sample, cache)

    @typing.overload
    def _cached_reports_implementation(
        self,
        cache_key: WorkdirCacheKey,
        fn: typing.Callable[[], dict[str, list[Path]]],
        sample: Optional[str] = None,
        cache: bool = True,
    ) -> list[Path]: ...

    @typing.overload
    def _cached_reports_implementation(
        self,
        cache_key: WorkdirCacheKey,
        fn: typing.Callable[[], dict[str, Path]],
        sample: Optional[str] = None,
        cache: bool = True,
    ) -> Path: ...

    def _cached_reports_implementation(
        self,
        cache_key: WorkdirCacheKey,
        fn: typing.Callable[[], dict[str, Path] | dict[str, list[Path]]],
        sample: Optional[str] = None,
        cache: bool = True,
    ) -> Path | list[Path]:
        if not cache or self._cache.get(cache_key) is None:
            self._cache[cache_key] = fn()

        if sample:
            res = self._cache[cache_key].get(sample)
            if res is None:
                raise WorkdirOutputNotFound(
                    sample_id=sample,
                    pattern=self._SEARCH_PATTERNS[cache_key],
                    workdir=self.basedir,
                )
            return res

        return list(flatten(self._cache[cache_key].values()))

    @typing.overload
    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: None = None,
        *,
        cache: bool = True,
    ) -> List[Path]: ...

    @typing.overload
    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: str,
        *,
        cache: bool = True,
    ) -> Path: ...

    def single_cell_report(
        self,
        stage: SingleCellStage | SingleCellStageLiteral,
        sample: Optional[str] = None,
        *,
        cache: bool = True,
    ) -> Path | List[Path]:
        """Retrieve a report from a stage of the single-cell pipeline.

        :param stage: The name of the stage to retrieve a report from
        :param sample: The sample name to filter on
        :param cache: Whether to return cached result if available
        :return: A list of metadata files
        """
        stage_key = stage.value if isinstance(stage, enum.Enum) else stage
        cache_key: WorkdirCacheKey = SINGLE_CELL_STAGES_TO_CACHE_KEY_MAPPING[stage_key]
        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)

    @typing.overload
    def filtered_dataset(
        self,
        sample: None = None,
        cache: bool = True,
    ) -> list[Path]: ...

    @typing.overload
    def filtered_dataset(
        self,
        sample: str,
        cache: bool = True,
    ) -> Path: ...

    def filtered_dataset(
        self, sample: str | None = None, cache: bool = True
    ) -> Path | list[Path]:
        """Return the path to a filtered dataset for a sample.

        This is the output `dataset.pxl` file from single-cell annotate

        :param sample: The sample name to retrieve the filtered dataset for
        :param cache: Whether to return cached result if available
        :return: The path to the filtered dataset if a sample is given, otherwise a list of all datasets
        """
        cache_key: WorkdirCacheKey = "single-cell-mpx annotate dataset"
        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)

    @typing.overload
    def raw_component_metrics(
        self,
        sample: None = None,
        cache: bool = True,
    ) -> list[Path]: ...

    @typing.overload
    def raw_component_metrics(
        self,
        sample: str,
        cache: bool = True,
    ) -> Path: ...

    def raw_component_metrics(
        self, sample: Optional[str] = None, cache: bool = True
    ) -> Path | list[Path]:
        """Return the path to a raw_component_metrics file.

        This is generated by single-cell annotate

        :param sample: The sample name to retrieve the filtered dataset for
        :param cache: Whether to return cached results if available
        :return: The path to a raw_component_metrics file if a sample is given, otherwise a list of all files
        """
        cache_key: WorkdirCacheKey = "single-cell-mpx annotate raw_components_metrics"
        func = functools.partial(self._collect_output_files, cache_key)
        return self._cached_reports_implementation(cache_key, func, sample, cache)
