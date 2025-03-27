"""Collect files and transform data for reporting from a :py:class:`PixelatorWorkdir`.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Iterable

import pandas as pd

from pixelator.mpx.report.common.cli_info import CLIInvocationInfo
from pixelator.mpx.report.common.workdir import (
    PixelatorWorkdir,
    SingleCellStage,
    WorkdirOutputNotFound,
)
from pixelator.mpx.report.models import (
    AdapterQCSampleReport,
    AmpliconSampleReport,
    AnalysisSampleReport,
    AnnotateSampleReport,
    CollapseSampleReport,
    CommandInfo,
    DemuxSampleReport,
    GraphSampleReport,
    MoleculesDataflowReport,
    PreQCSampleReport,
    ReadsDataflowReport,
)
from pixelator.mpx.report.models.base import SampleReport
from pixelator.mpx.report.models.layout import LayoutSampleReport

logger = logging.getLogger("pixelator.report")
ModelT = typing.TypeVar("ModelT", bound=type[SampleReport])


class PixelatorReporting:
    """Collect files for reporting from a :py:class:`PixelatorWorkdir`.

    :ivar workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
        instance
    """

    def __init__(self, workdir: Path | PixelatorWorkdir):
        """Initialize the PixelatorReporting object.

        :param workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
        """
        if isinstance(workdir, PixelatorWorkdir):
            self.workdir = workdir
        else:
            self.workdir = PixelatorWorkdir(workdir)

        # all pixelator commands in defined order
        self._command_list: list[str] | None = None

    def samples(self) -> set[str]:
        """Return a list of all samples encountered from command metadata files."""
        return self.workdir.samples()

    @staticmethod
    def _combine_data(reports, model: ModelT) -> pd.DataFrame:
        """Merge a list of :class:`Model` objects into a single dataframe."""
        data = []

        for r in reports:
            logger.debug("Parsing metrics file: %s", r)
            data.append(model.from_json(r).model_dump())

        df = pd.DataFrame(data)
        df.astype({"sample_id": "string"})
        df.set_index("sample_id", inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _explode_json_columns(
        df: pd.DataFrame, keys_to_explode: Iterable[str]
    ) -> pd.DataFrame:
        """Explode json columns of a dataframe.

        :param df: The dataframe to explode
        :param keys_to_explode: The keys of the json columns to explode
        """
        series_to_concat = []
        for key in keys_to_explode:
            dict_column = pd.json_normalize(df[key])
            cleaned_key = key.replace("_stats", "")
            dict_column.columns = [
                f"{cleaned_key}_{col}" for col in dict_column.columns
            ]
            dict_column.set_index(df.index, inplace=True)
            series_to_concat.append(dict_column)

        # Merge the exploded dict into the original dataframe and drop the dict column
        df.drop(
            labels=list(keys_to_explode),
            axis="columns",
            inplace=True,
        )
        df = pd.concat(
            (df, *series_to_concat),
            join="inner",
            axis="columns",
        )
        return df

    def amplicon_metrics(self, sample_name: str) -> AmpliconSampleReport:
        """Return the amplicon metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.AMPLICON, sample_name
        )
        return AmpliconSampleReport.from_json(sample_file)

    def amplicon_summary(self) -> pd.DataFrame:
        """Combine all amplicon reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.AMPLICON)
        df = self._combine_data(reports, AmpliconSampleReport)
        return df

    def preqc_metrics(self, sample_name: str) -> PreQCSampleReport:
        """Return the preqc metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.PREQC, sample_name
        )
        return PreQCSampleReport.from_json(sample_file)

    def preqc_summary(self) -> pd.DataFrame:
        """Combine all preqc reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.PREQC)
        df = self._combine_data(reports, PreQCSampleReport)
        return df

    def adapterqc_metrics(self, sample_name: str) -> AdapterQCSampleReport:
        """Return the adapterqc metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ADAPTERQC, sample_name
        )
        return AdapterQCSampleReport.from_json(sample_file)

    def adapterqc_summary(self) -> pd.DataFrame:
        """Combine all preqc reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ADAPTERQC)
        df = self._combine_data(reports, AdapterQCSampleReport)
        return df

    def demux_metrics(self, sample_name: str) -> DemuxSampleReport:
        """Return the demux metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.DEMUX, sample_name
        )
        return DemuxSampleReport.from_json(sample_file)

    def demux_summary(self) -> pd.DataFrame:
        """Combine all demux reports into a single dataframe.

        For example:
                            input_read_count  output_read_count ACTB ...  mIgG2b
        sample_id                                                   55 ...      89
        pbmcs_unstimulated             199390              189009   86 ...     115
        """
        reports = self.workdir.single_cell_report(SingleCellStage.DEMUX)
        df = self._combine_data(reports, DemuxSampleReport)

        # We do not use _explode_json_columns here because the
        # keys naming is different here

        # Extract the dict column from the dataframe and expand each key
        # into a new column
        read_dict = pd.json_normalize(df["per_antibody_read_counts"])
        read_dict.columns = [f"read_counts_{col}" for col in read_dict.columns]
        read_dict.set_index(df.index, inplace=True)

        read_fraction_dict = pd.json_normalize(df["per_antibody_read_count_fractions"])
        read_fraction_dict.columns = [
            f"read_count_fraction_{col}" for col in read_fraction_dict.columns
        ]
        read_fraction_dict.set_index(df.index, inplace=True)

        # Merge the exploded dict into the original dataframe and drop the dict column
        df.drop(
            labels=["per_antibody_read_counts", "per_antibody_read_count_fractions"],
            axis="columns",
            inplace=True,
        )
        df = pd.concat(
            (df, read_dict, read_fraction_dict), join="inner", axis="columns"
        )
        return df

    def collapse_metrics(self, sample_name: str) -> CollapseSampleReport:
        """Return the collapse metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.COLLAPSE, sample_name
        )
        return CollapseSampleReport.from_json(sample_file)

    def collapse_summary(self) -> pd.DataFrame:
        """Combine all collapse sumaries into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.COLLAPSE)
        df = self._combine_data(reports, CollapseSampleReport)

        keys_to_explode = ["collapsed_molecule_count_stats"]
        df = self._explode_json_columns(df, keys_to_explode)
        return df

    def graph_metrics(self, sample_name: str) -> GraphSampleReport:
        """Return the graph metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.GRAPH, sample_name
        )
        return GraphSampleReport.from_json(sample_file)

    def graph_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.GRAPH)
        df = self._combine_data(reports, GraphSampleReport)

        # Extract the dict column from the dataframe and expand each key
        # into a new column
        keys_to_explode = [
            "read_count_per_molecule_stats",
        ]

        df = self._explode_json_columns(df, keys_to_explode)
        return df

    def annotate_metrics(self, sample_name: str) -> AnnotateSampleReport:
        """Return the annotate metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ANNOTATE, sample_name
        )
        return AnnotateSampleReport.from_json(sample_file)

    def annotate_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ANNOTATE)
        df = self._combine_data(reports, AnnotateSampleReport)

        keys_to_explode = [
            "molecule_count_per_cell_stats",
            "read_count_per_cell_stats",
            "a_pixel_count_per_cell_stats",
            "b_pixel_count_per_cell_stats",
            "marker_count_per_cell_stats",
            "a_pixel_b_pixel_ratio_per_cell_stats",
            "molecule_count_per_a_pixel_per_cell_stats",
            "b_pixel_count_per_a_pixel_per_cell_stats",
            "a_pixel_count_per_b_pixel_per_cell_stats",
        ]

        df = self._explode_json_columns(df, keys_to_explode)
        return df

    def layout_metrics(self, sample_name: str) -> LayoutSampleReport:
        """Return the layout metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.LAYOUT, sample_name
        )
        return LayoutSampleReport.from_json(sample_file)

    def layout_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.LAYOUT)
        df = self._combine_data(reports, LayoutSampleReport)
        return df

    def analysis_metrics(self, sample_name: str) -> AnalysisSampleReport:
        """Return the analysis metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ANALYSIS, sample_name
        )
        return AnalysisSampleReport.from_json(sample_file)

    def analysis_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ANALYSIS)
        df = self._combine_data(reports, AnalysisSampleReport)
        keys_to_explode = ["polarization", "colocalization"]
        df = self._explode_json_columns(df, keys_to_explode)
        return df

    def reads_flow(self, sample_name: str) -> ReadsDataflowReport:
        """Return a summary with the flow of read counts through the pipeline.

        :param sample_name: The sample to return the dataflow for
        :return ReadsDataflowReport: A class:`ReadsDataflowReport` instance
        """
        preqc_metrics = self.preqc_metrics(sample_name)
        adapterqc_metrics = self.adapterqc_metrics(sample_name)
        demux_metrics = self.demux_metrics(sample_name)
        collapse_metrics = self.collapse_metrics(sample_name)
        annotate_metrics = self.annotate_metrics(sample_name)

        return ReadsDataflowReport(
            sample_id=sample_name,
            input_read_count=preqc_metrics.total_read_count,
            qc_filtered_read_count=preqc_metrics.passed_filter_read_count,
            valid_pbs_read_count=adapterqc_metrics.passed_filter_read_count,
            valid_antibody_read_count=demux_metrics.output_read_count,
            raw_molecule_read_count=collapse_metrics.output_read_count,
            size_filter_fail_molecule_read_count=annotate_metrics.size_filter_fail_read_count,
            aggregate_molecule_read_count=annotate_metrics.reads_in_aggregates_count,
            cell_molecule_read_count=annotate_metrics.read_count,
        )

    def reads_flow_summary(self):
        """Combine reads flow metrics for all samples into a single dataframe."""
        data = [self.reads_flow(sample).model_dump() for sample in self.samples()]

        df = pd.DataFrame(data)
        df.astype({"sample_id": "string"})
        df.set_index("sample_id", inplace=True)
        df.sort_index(inplace=True)

        return df

    def molecules_flow(self, sample_name: str) -> MoleculesDataflowReport:
        """Return a summary with the flow of molecules counts through the pipeline.

        :param sample_name: The sample to return the dataflow for
        :return MoleculesDataflowReport:
            A class:`MoleculesDataflowReport` instance
        """
        collapse_metrics = self.collapse_metrics(sample_name)
        annotate_metrics = self.annotate_metrics(sample_name)

        return MoleculesDataflowReport(
            sample_id=sample_name,
            raw_molecule_count=collapse_metrics.molecule_count,
            size_filter_fail_molecule_count=annotate_metrics.size_filter_fail_molecule_count,
            aggregate_molecule_count=annotate_metrics.molecules_in_aggregates_count,
            cell_molecule_count=annotate_metrics.molecule_count,
        )

    def molecules_flow_summary(self):
        """Combine molecule flow metrics for all samples into a single dataframe."""
        data = [self.molecules_flow(sample).model_dump() for sample in self.samples()]

        df = pd.DataFrame(data)
        df.astype({"sample_id": "string"})
        df.set_index("sample_id", inplace=True)
        df.sort_index(inplace=True)
        return df

    def cli_invocation_info(self, sample: str, cache: bool = True) -> CLIInvocationInfo:
        """Return the commandline options used to invoke multiple pixelator commands on a sample.

        :param sample: The sample to return the commandline for
        :param cache: Use cached data if available
        :return: A `CLIInvocationInfo` to query commandline invocations.
        :raises WorkdirOutputNotFound: If no commandline metadata is found for the sample
        """
        # Function scope import to avoid circular dependencies

        try:
            metadata_files = self.workdir.metadata_files(sample, cache=cache)
        except WorkdirOutputNotFound as e:
            e.message = f'No command line metadata found for sample: "{sample}"'
            raise

        data_flat = []
        for f in metadata_files:
            command_info_flat = CommandInfo.from_json(f)
            data_flat.append(command_info_flat)

        if self._command_list is None:
            self._command_list = _ordered_pixelator_commands()

        # mypy cannot detect that self._command_list cannot be None
        # anymore at this point so we cast it explicitly.
        data_flat.sort(
            key=lambda x: typing.cast("list[str]", self._command_list).index(x.command)
        )
        return CLIInvocationInfo(data_flat, sample_id=sample)


def _ordered_pixelator_commands() -> list[str]:
    """Return a list of pixelator CLI commands in depth-first order."""
    # local imports to avoid circular dependency issues
    import click

    from pixelator.cli import main_cli as click_context

    def build_command_list(obj: typing.Any, prefix=None):
        """Recursively go through groups and commands to create a list of commands."""
        prefix = prefix or []

        if isinstance(obj, click.Group):
            res = []
            new_prefix = prefix + [obj.name]

            for subcommand in obj.commands.values():
                res.extend(build_command_list(subcommand, prefix=new_prefix))

            return res

        return (" ".join(prefix + [obj.name]),)

    return build_command_list(click_context)
