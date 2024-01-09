"""Functions to create a summary interactive web report.

Copyright (c) 2022 Pixelgen Technologies AB.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional  # noqa: F401

import numpy as np
import pandas as pd

# Any is used by type hint comment
from anndata import AnnData

from pixelator import __version__
from pixelator.pixeldataset import PixelDataset
from pixelator.report.models.report_metadata import SampleMetadata
from pixelator.report.qcreport import (
    Metrics,
    SampleInfo,
    QCReportBuilder,
    collect_report_data,
)

from pixelator.report.common import PixelatorWorkdir, PixelatorReporting
from pixelator.utils import get_sample_name

logger = logging.getLogger(__name__)

DEFAULT_METRICS_DEFINITION_FILE = Path(__file__).parent / "webreport/metrics.json"


def create_dynamic_report(
    reporting: PixelatorReporting,
    sample_name: str,
    info: SampleInfo,
    output_path: str,
) -> None:
    """Create a dynamics qc report of a single sample.

    A helper function to create a dynamic web report of a single sample. The
    function uses a template stored in 'webreport/template.html' as a base
    template which is filled with the different metrics and stats.

    :param input_path: the path to results folder containing all the steps
    :param reporting: A PixelatorReporting object
    :param sample_name: the name of the sample
    :param info: A dictionary containing meta information about the sample
    :param output_path: the path to the output folder
    :returns: None
    :rtype: None
    """
    sample_id = info.sample_id
    logger.info("Creating QC report for sample %s", sample_id)

    amplicon_metrics = reporting.amplicon_metrics(sample_name)
    preqc_metrics = reporting.preqc_metrics(sample_name)
    demux_metrics = reporting.demux_metrics(sample_name)
    collapse_metrics = reporting.collapse_metrics(sample_name)
    graph_metrics = reporting.graph_metrics(sample_name)
    annotate_metrics = reporting.annotate_metrics(sample_name)
    flow_metrics = reporting.reads_and_molecules_flow(sample_name)

    # Collect antibody metrics
    antibodies_data_values = {
        "antibody_reads": demux_metrics.output_read_count,
        "antibody_reads_usable_per_cell": annotate_metrics.total_reads_cell,
        "antibody_reads_in_outliers": annotate_metrics.reads_of_aggregates,
        "unrecognized_antibodies": demux_metrics.unrecognised_antibody_read_count,
    }

    antibodies_data_fractions = (
        pd.Series(antibodies_data_values) / flow_metrics.input_read_count
    ).to_dict()

    antibodies_data_fractions = {
        f"fraction_{k}": v for k, v in antibodies_data_fractions.items()
    }

    placeholder_cell_predictions = {
        "predicted_cell_type_b_cells": None,
        "fraction_predicted_cell_type_b_cells": None,
        "predicted_cell_type_cd4p_cells": None,
        "fraction_predicted_cell_type_cd4p_cells": None,
        "predicted_cell_type_cd8p_cells": None,
        "fraction_predicted_cell_type_cd8p_cells": None,
        "predicted_cell_type_monocytes": None,
        "fraction_predicted_cell_type_monocytes": None,
        "predicted_cell_type_nk_cells": None,
        "fraction_predicted_cell_type_nk_cells": None,
        "predicted_cell_type_unknown": None,
        "fraction_predicted_cell_type_unknown": None,
    }

    # The total number of input reads per cell.
    # Note that this is not the number of actual reads that were assigned to the cell.
    # This gives an idea of how many reads are lost in the pipeline averaged over the number of cell
    average_input_read_per_cell = (
        flow_metrics.input_read_count / annotate_metrics.cells_filtered
    )

    fraction_reads_in_cells = (
        annotate_metrics.total_reads_cell / flow_metrics.input_read_count
    )

    # Map pixelator metrics to QC report metrics
    metrics = Metrics(
        number_of_cells=annotate_metrics.cells_filtered,
        average_reads_usable_per_cell=int(annotate_metrics.mean_reads_cell),
        average_reads_per_cell=int(average_input_read_per_cell),
        average_antibody_molecules_per_cell=annotate_metrics.mean_umi_cell,
        average_upias_per_cell=int(annotate_metrics.mean_upia_cell),
        average_umis_per_upia=int(annotate_metrics.mean_umi_upia_cell),
        fraction_reads_in_cells=fraction_reads_in_cells,
        fraction_discarded_umis=annotate_metrics.fraction_umis_in_non_cell_components,
        total_unique_antibodies_detected=annotate_metrics.total_markers,
        number_of_reads=flow_metrics.input_read_count,
        number_of_short_reads_skipped=preqc_metrics.too_short_read_count,
        fraction_valid_pbs=flow_metrics.fraction_valid_pbs_reads,
        fraction_valid_umis=flow_metrics.fraction_valid_umi_reads,
        average_reads_per_molecule=graph_metrics.mean_count,
        sequencing_saturation=collapse_metrics.fraction_duplicate_reads,
        fraction_q30_bases_in_antibody_barcode=amplicon_metrics.fraction_q30_bc,
        fraction_q30_bases_in_umi=amplicon_metrics.fraction_q30_umi,
        fraction_q30_bases_in_upia=amplicon_metrics.fraction_q30_upia,
        fraction_q30_bases_in_upib=amplicon_metrics.fraction_q30_upib,
        fraction_q30_bases_in_pbs1=amplicon_metrics.fraction_q30_pbs1,
        fraction_q30_bases_in_pbs2=amplicon_metrics.fraction_q30_pbs2,
        fraction_q30_bases_in_read=amplicon_metrics.fraction_q30,
        **antibodies_data_values,  # type: ignore
        **antibodies_data_fractions,  # type: ignore
        **placeholder_cell_predictions,  # type: ignore
    )

    data = collect_report_data(reporting.workdir, sample_id)

    # uses the default template
    builder = QCReportBuilder()

    report_path = str(Path(output_path) / f"{sample_id}.qc-report.html")
    with open(report_path, "wb") as f:
        builder.write(f, sample_info=info, metrics=metrics, data=data)

    logger.debug("Dynamic webreport created in %s", output_path)


def make_report(
    input_path: str,
    output_path: str,
    panel: Optional[str],
    metadata: Optional[str],
    verbose: Optional[bool],
) -> None:
    """Parse every stage metrics and interactive plots to a single HTML file.

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
    from pixelator.config import config, load_antibody_panel

    logger.info("Generating QC reports from %s", input_path)

    # TODO: Move file collection and workdir scanning
    #       logic of *_metrics functions to PixelatorWorkdir

    # collect and save metrics
    if panel is not None:
        panel_obj = load_antibody_panel(config, panel)
    else:
        panel_obj = None

    reporting = PixelatorReporting(Path(input_path))

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
        pixel_version = ""

        try:
            collapse_design_option = invocation_info.get_option("collapse", "--design")
            pixel_version = collapse_design_option.value
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
            generation_date=datetime.utcnow().isoformat(),
            sample_id=sample,
            sample_description=sample_description,
            pixel_version=pixel_version,
            panel_name=webreport_panel_name,
            panel_version=webreport_panel_version,
            parameters=list(invocation_info),
        )

        create_dynamic_report(
            reporting,
            sample_name=sample,
            info=sample_info,
            output_path=output_path,
        )

    logger.info("Finished generating QC reports")
