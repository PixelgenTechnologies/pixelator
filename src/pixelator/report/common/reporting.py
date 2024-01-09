from __future__ import annotations

from pathlib import Path

import pandas as pd

from pixelator.report import PixelatorWorkdir
from pixelator.report.models import (
    AdapterQCStageReport,
    AmpliconStageReport,
    AnalysisStageReport,
    AnnotateStageReport,
    CollapseStageReport,
    CommandInfo,
    DemuxStageReport,
    GraphStageReport,
    PreQCStageReport,
    ReadsAndMoleculesDataflowReport,
)
from pixelator.report.workdir import Model, SingleCellStage, logger


class PixelatorReporting:
    """Collect files for reporting from a :py:class:`PixelatorWorkdir`.

    :param workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
        instance
    """

    def __init__(self, workdir: Path | PixelatorWorkdir):
        if isinstance(workdir, PixelatorWorkdir):
            self.workdir = workdir
        else:
            self.workdir = PixelatorWorkdir(workdir)

    def samples(self) -> set[str]:
        """Return a list of all samples encountered from command metadata files."""
        return self.workdir.samples()

    @staticmethod
    def _process_data(report, model: Model) -> Model:
        logger.debug("Parsing metrics file: %s", report)
        return model.from_json(report)

    @staticmethod
    def _combine_data(reports, model: Model) -> pd.DataFrame:
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

    def amplicon_metrics(self, sample_name: str) -> AmpliconStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.AMPLICON, sample_name
        )
        return AmpliconStageReport.from_json(sample_file)

    def amplicon_summary(self) -> pd.DataFrame:
        """Combine all amplicon reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.AMPLICON)
        df = self._combine_data(reports, AmpliconStageReport)
        return df

    def preqc_metrics(self, sample_name: str) -> PreQCStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.PREQC, sample_name
        )
        return PreQCStageReport.from_json(sample_file)

    def preqc_summary(self) -> pd.DataFrame:
        """Combine all preqc reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.PREQC)
        df = self._combine_data(reports, PreQCStageReport)
        return df

    def adapterqc_metrics(self, sample_name: str) -> AdapterQCStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ADAPTERQC, sample_name
        )
        return AdapterQCStageReport.from_json(sample_file)

    def adapterqc_summary(self) -> pd.DataFrame:
        """Combine all preqc reports into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ADAPTERQC)
        df = self._combine_data(reports, AdapterQCStageReport)
        return df

    def demux_metrics(self, sample_name: str) -> DemuxStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.DEMUX, sample_name
        )
        return DemuxStageReport.from_json(sample_file)

    def demux_summary(self) -> pd.DataFrame:
        """Combine all demux sumaries into a single dataframe.

        For example:
                            input_read_count  output_read_count ACTB ...  mIgG2b
        sample_id                                                   55 ...      89
        pbmcs_unstimulated             199390              189009   86 ...     115
        """
        reports = self.workdir.single_cell_report(SingleCellStage.DEMUX)
        df = self._combine_data(reports, DemuxStageReport)

        # Extract the dict column from the dataframe and expand each key
        # into a new column
        explode_dict = pd.json_normalize(df["per_antibody_read_counts"])
        explode_dict.set_index(df.index, inplace=True)

        # Merge the exploded dict into the original dataframe and drop the dict column
        df.drop("per_antibody_read_counts", axis="columns", inplace=True)
        df = pd.concat((df, explode_dict), join="inner", axis="columns")
        return df

    def collapse_metrics(self, sample_name: str) -> CollapseStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.COLLAPSE, sample_name
        )
        return CollapseStageReport.from_json(sample_file)

    def collapse_summary(self) -> pd.DataFrame:
        """Combine all collapse sumaries into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.COLLAPSE)
        df = self._combine_data(reports, CollapseStageReport)
        return df

    def graph_metrics(self, sample_name: str) -> GraphStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.GRAPH, sample_name
        )
        return GraphStageReport.from_json(sample_file)

    def graph_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.GRAPH)
        df = self._combine_data(reports, GraphStageReport)
        return df

    def annotate_metrics(self, sample_name: str) -> AnnotateStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ANNOTATE, sample_name
        )
        return AnnotateStageReport.from_json(sample_file)

    def annotate_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ANNOTATE)
        df = self._combine_data(reports, AnnotateStageReport)
        return df

    def analysis_metrics(self, sample_name: str) -> AnalysisStageReport:
        sample_file = self.workdir.single_cell_report(
            SingleCellStage.ANALYSIS, sample_name
        )
        return AnalysisStageReport.from_json(sample_file)

    def analysis_summary(self) -> pd.DataFrame:
        """Combine graph metrics for all samples into a single dataframe."""
        reports = self.workdir.single_cell_report(SingleCellStage.ANALYSIS)
        df = self._combine_data(reports, AnalysisStageReport)
        return df

    def reads_and_molecules_flow(
        self, sample_name: str
    ) -> ReadsAndMoleculesDataflowReport:
        preqc_metrics = self.preqc_metrics(sample_name)
        adapterqc_metrics = self.adapterqc_metrics(sample_name)
        demux_metrics = self.demux_metrics(sample_name)
        collapse_metrics = self.collapse_metrics(sample_name)
        annotate_metrics = self.annotate_metrics(sample_name)

        return ReadsAndMoleculesDataflowReport(
            sample_id=sample_name,
            input_read_count=preqc_metrics.total_read_count,
            qc_filtered_read_count=preqc_metrics.passed_filter_read_count,
            valid_pbs_read_count=adapterqc_metrics.passed_filter_read_count,
            valid_antibody_read_count=demux_metrics.output_read_count,
            unique_molecule_read_count=collapse_metrics.output_read_count,
            unique_molecule_count=collapse_metrics.unique_molecule_count,
            unique_molecule_in_cells_read_count=annotate_metrics.total_reads_cell,
            unique_molecule_in_cells_count=annotate_metrics.cells_filtered,
        )

    def cli_invocation_info(self, sample: str) -> CLIInvocationInfo:
        """Return the commandline options used to invoke multiple pixelator commands on a sample.

        :param sample: The sample to return the commandline for
        """
        # Function scope import to avoid circular dependencies
        from pixelator.cli import main_cli as click_context

        metadata_files = self.workdir.metadata_files(sample)

        if len(metadata_files) == 0:
            raise KeyError(f"No commandline metadata found for sample")

        data_flat: list[CommandInfo] = []
        order = list(click_context.commands["single-cell"].commands.keys())  # type: ignore

        for f in metadata_files:
            command_info_flat = CommandInfo.from_json(f)
            data_flat.append(command_info_flat)

        data_flat.sort(key=lambda x: order.index(x.command.split(" ")[-1]))
        return CLIInvocationInfo(data_flat, sample_id=sample)
