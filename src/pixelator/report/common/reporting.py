"""Copyright © 2023 Pixelgen Technologies AB."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

import pandas as pd

from pixelator.report.common.cli_info import CLIInvocationInfo
from pixelator.report.common.workdir import (
    WorkdirOutputNotFound,
    PixelatorWorkdir,
    SingleCellStage,
)
from pixelator.report.models import (
    AdapterQCSampleReport,
    AmpliconSampleReport,
    AnalysisSampleReport,
    AnnotateSampleReport,
    CollapseSampleReport,
    CommandInfo,
    DemuxSampleReport,
    GraphSampleReport,
    PreQCSampleReport,
    ReadsDataflowReport,
    MoleculesDataflowReport,
)
from pixelator.report.models.base import SampleReport

logger = logging.getLogger("pixelator.report")
ModelT = typing.TypeVar("ModelT", bound=type[SampleReport])


class PixelatorReporting:
    """Collect files for reporting from a :py:class:`PixelatorWorkdir`.

    :param workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
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
        read_count_per_molecule_dict = pd.json_normalize(
            df["read_count_per_molecule_stats"]
        )
        read_count_per_molecule_dict.columns = [
            f"read_count_per_molecule_{col}"
            for col in read_count_per_molecule_dict.columns
        ]
        read_count_per_molecule_dict.set_index(df.index, inplace=True)

        molecule_count_per_a_pixel_dict = pd.json_normalize(
            df["molecule_count_per_a_pixel_stats"]
        )
        molecule_count_per_a_pixel_dict.columns = [
            f"molecule_count_per_a_pixel_{col}"
            for col in molecule_count_per_a_pixel_dict.columns
        ]
        molecule_count_per_a_pixel_dict.set_index(df.index, inplace=True)

        b_pixel_count_per_a_pixel_dict = pd.json_normalize(
            df["b_pixel_count_per_a_pixel_stats"]
        )
        b_pixel_count_per_a_pixel_dict.columns = [
            f"b_pixel_count_per_a_pixel_{col}"
            for col in b_pixel_count_per_a_pixel_dict.columns
        ]
        b_pixel_count_per_a_pixel_dict.set_index(df.index, inplace=True)

        # Merge the exploded dict into the original dataframe and drop the dict column
        df.drop(
            labels=[
                "read_count_per_molecule_stats",
                "molecule_count_per_a_pixel_stats",
                "b_pixel_count_per_a_pixel_stats",
            ],
            axis="columns",
            inplace=True,
        )
        df = pd.concat(
            (
                df,
                read_count_per_molecule_dict,
                molecule_count_per_a_pixel_dict,
                b_pixel_count_per_a_pixel_dict,
            ),
            join="inner",
            axis="columns",
        )
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

        # Extract the dict column from the dataframe and expand each key
        # into a new column
        polarization_dict = pd.json_normalize(df["polarization"])
        polarization_dict.set_index(df.index, inplace=True)

        colocalization_dict = pd.json_normalize(df["colocalization"])
        polarization_dict.set_index(df.index, inplace=True)

        # Merge the exploded dict into the original dataframe and drop the dict column
        df.drop("polarization", axis="columns", inplace=True)
        df.drop("colocalization", axis="columns", inplace=True)

        df = pd.concat(
            (df, polarization_dict, colocalization_dict), join="inner", axis="columns"
        )
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

    def molecules_flow(self, sample_name: str) -> MoleculesDataflowReport:
        """Return a summary with the flow of molecules counts through the pipeline.

        :param sample_name: The sample to return the dataflow for
        :return ReadsAndMoleculesDataflowReport: A class:`ReadsAndMoleculesDataflowReport` instance
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

    def cli_invocation_info(self, sample: str, cache: bool = True) -> CLIInvocationInfo:
        """Return the commandline options used to invoke multiple pixelator commands on a sample.

        :param sample: The sample to return the commandline for
        :param cache: Use cached data if available
        :return: A `CLIInvocationInfo` to query commandline invocations.
        :raises WorkdirOutputNotFound: If no commandline metadata is found for the sample
        """
        # Function scope import to avoid circular dependencies
        from pixelator.cli import main_cli as click_context

        try:
            metadata_files = self.workdir.metadata_files(sample, cache=cache)
        except WorkdirOutputNotFound as e:
            e.message = f'No command line metadata found for sample: "{sample}"'
            raise

        data_flat: list[CommandInfo] = []
        order = list(click_context.commands["single-cell"].commands.keys())  # type: ignore

        for f in metadata_files:
            command_info_flat = CommandInfo.from_json(f)
            data_flat.append(command_info_flat)

        data_flat.sort(key=lambda x: order.index(x.command.split(" ")[-1]))
        return CLIInvocationInfo(data_flat, sample_id=sample)
