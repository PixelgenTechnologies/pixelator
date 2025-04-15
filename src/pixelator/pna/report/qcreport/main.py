"""Functions to create a summary interactive web report.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pixelator import __version__
from pixelator.pna.report.common import PixelatorPNAReporting
from pixelator.pna.report.models.report_metadata import SampleMetadata
from pixelator.pna.report.qcreport.builder import PNAQCReportBuilder
from pixelator.pna.report.qcreport.collect import collect_report_data
from pixelator.pna.report.qcreport.types import SampleInfo

logger = logging.getLogger("pixelator.report")


def create_qc_report(
    reporting: PixelatorPNAReporting,
    sample_name: str,
    info: SampleInfo,
    output_path: Path,
) -> None:
    """Create a dynamic customer QC report for a single sample.

    A helper function to create a dynamic web report of a single sample.
    The function uses a template stored in 'report/qcreport/template.html'
    as a base template which is filled with the different metrics and stats.

    :param reporting: A PixelatorReporting object
    :param sample_name: the name of the sample
    :param info: A dictionary containing meta information about the sample
    :param output_path: the path to the output folder
    """
    sample_id = info.sample_id
    logger.info("Creating QC report for sample %s", sample_id)

    report_data = collect_report_data(reporting, sample_id)

    # uses the default template
    builder = PNAQCReportBuilder()
    report_path = str(Path(output_path) / f"{sample_id}.qc-report.html")

    with open(report_path, "wb") as f:
        builder.write(f, sample_info=info, data=report_data)

    logger.debug("QC report created in %s", output_path)


def create_per_sample_qc_reports(
    input_path: Path,
    output_path: Path,
    panel: Optional[str],
    metadata: Optional[str],
    verbose: Optional[bool],
) -> None:
    """Parse stage metrics and build an interactive HTML report file.

    This function iterates the different steps (preqc, adapterqc, demux, collapse,
    graph and annotate) for a given processed dataset (folder). The function will parse
    the metrics of all the samples present in each subfolder (step). These metrics will
    then be converted to tables. The function will also generate interactive plotly
    figures using the output data present in the graph/annotate steps. All the tables
    and figures will be saved to files and used to generate individual html report for
    each sample (web reports).

    :param input_path: the path to results folder containing all the steps
    :param output_path: the path to the output folder
    :param panel: a path to a panel file or a panel name to load from the config
    :param metadata: a path to a metadata file
    :param verbose: run if verbose mode when true
    :returns: None
    :rtype: None
    :raises AssertionError: when the input is not valid
    :raises RuntimeError: when the metadata is not valid
    """
    from pixelator.pna.config import load_antibody_panel, pna_config

    logger.info("Generating PNA QC reports from %s", input_path)

    # collect and save metrics
    if panel is not None:
        panel_obj = load_antibody_panel(pna_config, panel)
    else:
        panel_obj = None

    reporting = PixelatorPNAReporting(Path(input_path))

    sample_metadata = SampleMetadata.from_csv(metadata) if metadata else None

    # create the dynamic report (one per sample)

    for sample in reporting.samples():
        this_sample_metadata = (
            sample_metadata.get_by_id(sample) if sample_metadata else None
        )
        sample_description = (
            this_sample_metadata.description if this_sample_metadata else ""
        )

        invocation_info = reporting.cli_invocation_info(sample)
        technology_version = ""

        try:
            collapse_design_option = invocation_info.get_option("collapse", "--design")
            technology_version = collapse_design_option.value
        except KeyError:
            logger.warning("No design info found for sample %s", sample)

        webreport_panel_name = None
        webreport_panel_version = None

        if panel_obj is not None:
            webreport_panel_name = (
                panel_obj.name if panel_obj.name else str(panel_obj.filename)
            )
            webreport_panel_version = panel_obj.version

        # create SampleInfo object
        sample_info = SampleInfo(
            pixelator_version=__version__,
            generation_date=datetime.now(timezone.utc).isoformat(),
            sample_id=sample,
            sample_description=sample_description,
            technology="PNA",
            technology_version=technology_version,
            panel_name=webreport_panel_name,
            panel_version=webreport_panel_version,
            parameters=list(invocation_info),  # type: ignore
        )

        create_qc_report(
            reporting,
            sample_name=sample,
            info=sample_info,
            output_path=output_path,
        )

    logger.info("Finished generating QC reports")
