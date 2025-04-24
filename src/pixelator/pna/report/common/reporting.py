"""Collect files and transform data for reporting from a :py:class:`PixelatorWorkdir`.

Copyright © 2023 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import logging
import typing
from pathlib import Path

from pixelator.mpx.report.common.reporting import _ordered_pixelator_commands
from pixelator.pna.amplicon.report import AmpliconSampleReport
from pixelator.pna.analysis.report import AnalysisSampleReport
from pixelator.pna.collapse.independent import IndependentCollapseSampleReport
from pixelator.pna.collapse.report import CollapseSampleReport
from pixelator.pna.demux.report import DemuxSampleReport
from pixelator.pna.graph.report import GraphSampleReport
from pixelator.pna.layout.report import LayoutSampleReport
from pixelator.pna.report.common.cli_info import CLIInvocationInfo
from pixelator.pna.report.common.workdir import (
    PixelatorPNAWorkdir,
    SingleCellPNAStage,
    WorkdirOutputNotFound,
)
from pixelator.pna.report.models import (
    CommandInfo,
    ReadsDataflowReport,
)
from pixelator.pna.report.models.base import SampleReport

logger = logging.getLogger("pixelator.report")
ModelT = typing.TypeVar("ModelT", bound=type[SampleReport])


class PixelatorPNAReporting:
    """Collect files for reporting from a :py:class:`PixelatorWorkdir`.

    :ivar workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
        instance
    """

    def __init__(self, workdir: Path | PixelatorPNAWorkdir):
        """Initialize the PixelatorReporting object.

        :param workdir: The pixelator output folder or a :py:class:`PixelatorWorkdir`
        """
        if isinstance(workdir, PixelatorPNAWorkdir):
            self.workdir = workdir
        else:
            self.workdir = PixelatorPNAWorkdir(workdir)

        # all pixelator commands in defined order
        self._command_list: list[str] | None = None

    def samples(self) -> set[str]:
        """Return a list of all samples encountered from command metadata files."""
        return self.workdir.samples()

    def amplicon_metrics(self, sample_name: str) -> AmpliconSampleReport:
        """Return the amplicon metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.AMPLICON, sample_name
        )
        return AmpliconSampleReport.from_json(sample_file)

    def demux_metrics(self, sample_name: str) -> DemuxSampleReport:
        """Return the demux metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.DEMUX, sample_name
        )
        return DemuxSampleReport.from_json(sample_file)

    def collapse_metrics(
        self, sample_name: str
    ) -> CollapseSampleReport | IndependentCollapseSampleReport:
        """Return the collapse metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.COLLAPSE, sample_name
        )
        with open(sample_file) as f:
            data = json.load(f)

            if data["report_type"] == "collapse-independent":
                return IndependentCollapseSampleReport.from_json(sample_file)
            else:
                return CollapseSampleReport.from_json(sample_file)

    def graph_metrics(self, sample_name: str) -> GraphSampleReport:
        """Return the graph metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.GRAPH, sample_name
        )
        return GraphSampleReport.from_json(sample_file)

    def analysis_metrics(self, sample_name: str) -> AnalysisSampleReport:
        """Return the analysis metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.ANALYSIS, sample_name
        )
        return AnalysisSampleReport.from_json(sample_file)

    def layout_metrics(self, sample_name: str) -> LayoutSampleReport:
        """Return the layout metrics for a sample."""
        sample_file = self.workdir.single_cell_report(
            SingleCellPNAStage.LAYOUT, sample_name
        )
        return LayoutSampleReport.from_json(sample_file)

    def reads_flow(self, sample_name: str) -> ReadsDataflowReport:
        """Return a summary with the flow of read counts through the pipeline.

        :param sample_name: The sample to return the dataflow for
        :return ReadsDataflowReport: A class:`ReadsDataflowReport` instance
        """
        amplicon_metrics = self.amplicon_metrics(sample_name)
        demux_metrics = self.demux_metrics(sample_name)
        collapse_metrics = self.collapse_metrics(sample_name)
        graph_metrics = self.graph_metrics(sample_name)

        return ReadsDataflowReport(
            product_id="single-cell-pna",
            sample_id=sample_name,
            input_read_count=amplicon_metrics.input_reads,
            amplicon_output_read_count=amplicon_metrics.output_reads,
            demux_output_read_count=demux_metrics.output_reads,
            collapse_output_molecule_count=collapse_metrics.output_molecules,
            graph_output_molecule_count=graph_metrics.molecules_output,
        )

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
        try:
            data_flat.sort(
                key=lambda x: typing.cast("list[str]", self._command_list).index(
                    x.command
                )
            )
        except ValueError as e:
            # Do not fail if the command is not in the list,
            # just log a warning and continue.
            unknown_commands = {x.command for x in data_flat} - set(self._command_list)
            logger.warning(
                f"Unknown command in meta.json file: {','.join(unknown_commands)}"
            )

        return CLIInvocationInfo(data_flat, sample_id=sample)  # type: ignore[arg-type]
