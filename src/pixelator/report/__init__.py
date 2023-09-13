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
from pixelator.report.webreport import (
    Metrics,
    SampleInfo,
    WebreportBuilder,
    collect_report_data,
)
from pixelator.report.webreport.collect import (
    collect_parameter_info,
    index_parameter_info,
)
from pixelator.utils import get_sample_name

logger = logging.getLogger(__name__)

DEFAULT_METRICS_DEFINITION_FILE = Path(__file__).parent / "webreport/metrics.json"


def check_indexes(dataframes: List[pd.DataFrame]) -> bool:
    """Check if all indexes of a list of Pandas DataFrames are the same.

    :param dataframes: a list of Pandas DataFrames
    :returns: True if all indexes are the same, False otherwise
    :rtype: bool
    """
    if not dataframes:
        return False

    first_index = dataframes[0].index
    for df in dataframes[1:]:
        if not first_index.equals(df.index):
            return False

    return True


def amplicon_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `amplicon` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `amplicon` step (subfolder). The path to the results
    folder must be given and in this the amplicon subfolder should be present.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    step = "amplicon"
    logger.debug("Collecting %s metrics in %s", step, path)

    # check that the amplicon folder is present
    source_path = Path(path) / step
    if not source_path.is_dir():
        raise AssertionError(f"{step} folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run {step}?"
        )

    # parse the metrics per sample
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)

        # These values should already be float but due to a small bug in
        # c399899 (25-05-2023) they were converted to strings.
        # Force them to floats here to avoid having to run the whole pipeline
        # from amplicon.
        try:
            data = data["phred_result"]
            metrics.append(
                {
                    "fraction_q30_barcode": float(data["fraction_q30_bc"]),
                    "fraction_q30_upia": float(data["fraction_q30_upia"]),
                    "fraction_q30_upib": float(data["fraction_q30_upib"]),
                    "fraction_q30_umi": float(data["fraction_q30_umi"]),
                    "fraction_q30_PBS1": float(data["fraction_q30_pbs1"]),
                    "fraction_q30_PBS2": float(data["fraction_q30_pbs2"]),
                    "fraction_q30": float(data["fraction_q30"]),
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame(
        index=samples_processed,
        data=metrics,
    )

    logger.debug("Finish collecting amplicon metrics")
    return df.sort_index()


def preqc_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `preqc` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `preqc` step (subfolder). The path to the results
    folder must be given and in this the preqc subfolder should be present.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting preqc metrics in %s", path)

    # check that the preqc folder is present
    source_path = Path(path) / "preqc"
    if not source_path.is_dir():
        raise AssertionError(f"preqc folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run preqc?"
        )

    # parse the metrics per sample
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)
        try:
            n_reads = data["summary"]["before_filtering"]["total_reads"]
            data = data["filtering_result"]
            metrics.append(
                {
                    "total_reads": n_reads,
                    "passed_filter_reads": data["passed_filter_reads"],
                    "low_quality_reads": data["low_quality_reads"],
                    "too_many_N_reads": data["too_many_N_reads"],
                    "too_short_reads": data["too_short_reads"],
                    "too_long_reads": data["too_long_reads"],
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame(
        index=samples_processed,
        data=metrics,
    )
    df["discarded"] = round(1 - (df["passed_filter_reads"] / df["total_reads"]), 2)

    logger.debug("Finish collecting preqc metrics")
    return df.sort_index()


def adapterqc_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `adapterqc` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `adapterqc` step (subfolder). The path to the results
    folder must be given and in this the adapterqc subfolder should be present.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting adapterqc metrics in %s", path)

    # check that the adapterqc folder is present
    source_path = Path(path) / "adapterqc"
    if not source_path.is_dir():
        raise AssertionError(f"adapterqc folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*.report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run adapterqc?"
        )

    # parse the metrics per sample
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)
        try:
            metrics.append(
                {
                    "input": data["read_counts"]["input"],
                    "output": data["read_counts"]["output"],
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame(index=samples_processed, data=metrics)
    df["discarded"] = round(1 - (df["output"] / df["input"]), 2)

    logger.debug("Finish collecting adapterqc metrics")
    return df.sort_index()


def demux_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `demux` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `demux` step (subfolder). The path to the results
    folder must be given and in this the demux subfolder should be present.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting demux metrics in %s", path)

    # check that the demux folder is present
    source_path = Path(path) / "demux"
    if not source_path.is_dir():
        raise AssertionError(f"demux folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*.report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run demux?"
        )

    # parse the metrics per sample
    metrics = []  # type: List[Dict[str, Any]]
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)
        try:
            metrics_dict = {}
            metrics_dict["input"] = data["read_counts"]["input"]
            output = data["read_counts"]["read1_with_adapter"]
            metrics_dict["output"] = output
            # ensure the antibodies are always read in the same order
            for x in sorted(data["adapters_read1"], key=lambda x: x["name"]):
                metrics_dict[x["name"]] = round(x["total_matches"] / output, 6) * 100
            metrics.append(metrics_dict)
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame.from_records(metrics, index=samples_processed).fillna(0.0)
    df = df.round(decimals=2)
    df = df.astype({"input": int, "output": int})

    # sort columns by total count (left to right)
    first_cols = ["input", "output"]
    other_cols = df.columns.difference(first_cols)
    other_cols = df[other_cols].sum(axis=0).sort_values(ascending=False).index
    df = df.loc[:, first_cols + other_cols.tolist()]
    # de-fragment to avoid a PerformanceWarning with a fragmented data frame
    df = df.copy()

    # calculate the percentage of discarded molecules
    df["discarded"] = round(1 - (df["output"] / df["input"]), 2)

    logger.debug("Finish collecting demux metrics")
    return df.sort_index()


def collapse_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `collapse` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `collapse` step (subfolder). The path to the results
    folder must be given and in this the collapse subfolder should be present.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting collapse metrics in %s", path)

    # check that the collapse folder is present
    source_path = Path(path) / "collapse"
    if not source_path.is_dir():
        raise AssertionError(f"collapse folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*.report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run collapse?"
        )

    # parse the metrics per sample
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)
        try:
            metrics.append(
                {
                    "input": data["total_count"],
                    "output_edges": data["total_pixels"],
                    "output_umi": data["total_unique_umi"],
                    "output_upi": data["total_unique_upi"],
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame(
        index=samples_processed,
        data=metrics,
    )
    df["duplication"] = round(1 - (df["output_umi"] / df["input"]), 2)

    logger.debug("Finish collecting collapse metrics")
    return df.sort_index()


def graph_and_annotate_metrics(
    path: str,
    folder: Literal["graph", "annotate"],
) -> pd.DataFrame:
    """Parse the metrics from the `graph` or `annotate` step.

    Helper function to parse the metrics (JSON) of all the samples
    corresponding to the `graph` or `annotate` step (subfolder).

    The path to the results folder must be given and in this the
    subfolder `folder` should be present.

    Whether to parse `graph` or `annotate` metrics is controlled by the argument
    `folder`.

    A dataframe with the main metrics per sample is returned.

    :param path: a path to the results folder
    :param folder: which results to parse (graph or annotate)
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting %s metrics in %s", folder, path)

    # check that the graph/annotate folder is present
    source_path = Path(path) / folder
    if not source_path.is_dir():
        raise AssertionError(f"{folder} folder missing in {path}")

    # collect the metrics files
    files = list(source_path.glob("*.report.json"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No metrics files found in {source_path}. Did you run {folder}?"
        )

    # parse the metrics per sample
    cell_column = "components" if folder == "graph" else "cells"
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics file %s", file)

        with open(file) as json_file:
            data = json.load(json_file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)
        try:
            metrics.append(
                {
                    "upia": data["total_upia"],
                    "upib": data["total_upib"],
                    "umi": data["total_umi"],
                    "vertices": data["vertices"],
                    "edges": data["edges"],
                    cell_column: data["components"],
                    "markers": data["markers"],
                    "modularity": data["components_modularity"],
                    "frac_upib_upia": data["frac_upib_upia"],
                    "upia_degree_mean": data["upia_degree_mean"],
                    "upia_degree_median": data["upia_degree_median"],
                    "frac_largest_edges": data["frac_largest_edges"],
                    "frac_largest_vertices": data["frac_largest_vertices"],
                }
            )
        except KeyError as error:
            raise RuntimeError(
                f"Field {str(error)} missing in metrics file {file}"
            ) from error

    # create data frame
    df = pd.DataFrame(
        index=samples_processed,
        data=metrics,
    )

    logger.debug("Finish collecting %s metrics", folder)
    return df.sort_index()


def cell_calling_metrics(path: str) -> pd.DataFrame:
    """Parse the metrics from the `annotate` step.

    Helper function to parse the output data corresponding to the `annotate`
    step (subfolder) in order to generate a dataframe with useful metrics
    related to cell calling, filtering and annotation.

    The path to the results folder must be given and in this the `annotate`
    subfolder should be present.

    :param path: a path to the results folder
    :returns: a pd.DataFrame with the metrics
    :rtype: pd.DataFrame
    :raises AssertionError: if the input is not valid
    :raises RuntimeError: if a field is missing
    """
    logger.debug("Collecting cell calling metrics in %s", path)

    # check that the annotate folder is present
    source_path = Path(path) / "annotate"
    if not source_path.is_dir():
        raise AssertionError(f"annotate folder missing in {path}")

    # collect the dataset files (filtered)
    files = list(source_path.glob("*.dataset.pxl"))
    if files is None or len(files) == 0:
        raise AssertionError(
            f"No dataset files found in {source_path}. Did you run annotate?"
        )

    def _collect_metrics(adata: AnnData) -> Dict[str, float]:
        """Collect metrics from an AnnData object.

        A helper private function to collect metrics from an AnnData object.

        :param adata: the AnnData object
        :returns: a dictionary of different metrics
        :rtype: Dict[str, float]
        """
        metrics = {}
        metrics["cells_filtered"] = adata.n_obs
        metrics["total_markers"] = adata.n_vars
        metrics["total_reads_cell"] = adata.obs["reads"].sum()
        metrics["median_reads_cell"] = adata.obs["reads"].median()
        metrics["mean_reads_cell"] = adata.obs["reads"].mean()
        metrics["median_upi_cell"] = adata.obs["vertices"].median()
        metrics["mean_upi_cell"] = adata.obs["vertices"].mean()
        metrics["median_upia_cell"] = adata.obs["upia"].median()
        metrics["mean_upia_cell"] = adata.obs["upia"].mean()
        metrics["median_umi_cell"] = adata.obs["umi"].median()
        metrics["mean_umi_cell"] = adata.obs["umi"].mean()
        metrics["median_umi_upia_cell"] = adata.obs["median_umi_per_upia"].median()
        metrics["mean_umi_upia_cell"] = adata.obs["mean_umi_per_upia"].mean()
        metrics["median_upia_degree_cell"] = adata.obs["median_upia_degree"].median()
        metrics["mean_upia_degree_cell"] = adata.obs["mean_upia_degree"].mean()
        metrics["median_markers_cell"] = adata.obs["antibodies"].median()
        metrics["mean_markers_cell"] = adata.obs["antibodies"].mean()
        metrics["upib_per_upia"] = adata.obs["upib"].sum() / adata.obs["upia"].sum()

        metrics[
            "reads_of_aggregates"
        ] = 0  # This metric needs to be initialized for the webreport
        # Tau type will only be available if it has been added in the annotate step
        if "tau_type" in adata.obs:
            metrics["number_of_aggregates"] = np.sum(adata.obs["tau_type"] != "normal")
            metrics["fraction_of_aggregates"] = np.sum(
                adata.obs["tau_type"] != "normal"
            ) / len(adata.obs["tau_type"])
            metrics["reads_of_aggregates"] = (
                adata[adata.obs["tau_type"] != "normal"].obs["reads"].sum()
            )

        if "min_size_threshold" in adata.uns:
            metrics["minimum_size_threshold"] = adata.uns["min_size_threshold"]

        if "max_size_threshold" in adata.uns:
            metrics["max_size_threshold"] = adata.uns["max_size_threshold"]

        if "doublet_size_threshold" in adata.uns:
            metrics["doublet_size_threshold"] = adata.uns["doublet_size_threshold"]

        return metrics

    # collect metrics
    metrics = []
    samples_processed = []
    for file in files:
        logger.debug("Parsing metrics from dataset %s", file)

        clean_name = get_sample_name(file)
        samples_processed.append(clean_name)

        # parse dataset to get metrics
        dataset = PixelDataset.from_file(str(file))
        metrics_dict = _collect_metrics(dataset.adata)
        metrics.append(metrics_dict)

    # create a dataframe with the metrics
    df = pd.DataFrame.from_records(
        metrics,
        index=samples_processed,
    ).round(decimals=2)
    df = df.astype(
        {
            "cells_filtered": int,
            "total_markers": int,
        }
    )

    logger.debug("Finish collecting cell calling metrics")
    return df.sort_index()


def create_dynamic_report(
    input_path: str,
    summary_all: pd.Series,
    summary_amplicon: pd.Series,
    summary_preqc: pd.Series,
    summary_demux: pd.Series,
    summary_collapse: pd.Series,
    summary_annotate: pd.Series,
    summary_cell_calling: pd.Series,
    info: SampleInfo,
    output_path: str,
) -> None:
    """Create a dynamics web report of a single sample.

    A helper function to create a dynamic web report of a single sample. The
    function uses a template stored in 'webreport/template.html' as a base
    template which is filled with the different metrics and stats.

    :param input_path: the path to results folder containing all the steps
    :param summary_all: a pd.Series with the stage metrics of the sample
    :param summary_amplicon: a pd.Series with the `amplicon` stage
        metrics of the sample
    :param summary_preqc: a pd.Series with the `preqc` stage metrics of the sample
    :param summary_demux: a pd.Series with the `demux` stage metrics of the sample
    :param summary_collapse: a pd.Series with the `collapse` stage metrics
        of the sample
    :param summary_annotate: a pd.Series with the `annotate` metrics of the sample
    :param summary_cell_calling: a pd.Series with the per cell calling metrics
        of the sample
    :param info: A dictionary containing meta information about the sample
    :param output_path: the path to the output folder
    :returns: None
    :rtype: None
    """
    sample_id = info.sample_id
    logger.debug("Creating dynamic web report for sample %s", sample_id)

    # Collect antibody metrics
    antibodies_data_values = {
        "antibody_reads": summary_demux["output"],
        "antibody_reads_usable_per_cell": summary_cell_calling["total_reads_cell"],
        "antibody_reads_in_aggregates": summary_cell_calling["reads_of_aggregates"],
        "unrecognized_antibodies": summary_demux["input"] - summary_demux["output"],
    }

    antibodies_data_fractions = {
        "fraction_antibody_reads": antibodies_data_values["antibody_reads"]
        / summary_all["reads"],
        "fraction_antibody_reads_usable_per_cell": antibodies_data_values[
            "antibody_reads_usable_per_cell"
        ]
        / summary_all["reads"],
        "fraction_antibody_reads_in_aggregates": antibodies_data_values[
            "antibody_reads_in_aggregates"
        ]
        / summary_all["reads"],
        "fraction_unrecognized_antibodies": antibodies_data_values[
            "unrecognized_antibodies"
        ]
        / summary_all["reads"],
    }

    metrics = Metrics(
        number_of_cells=summary_cell_calling["cells_filtered"],
        average_reads_usable_per_cell=summary_cell_calling["mean_reads_cell"],
        average_reads_per_cell=(
            summary_all["reads"] / summary_cell_calling["cells_filtered"]
        ),
        median_antibody_molecules_per_cell=summary_cell_calling["median_umi_cell"],
        average_upis_per_cell=summary_cell_calling["mean_upia_cell"],
        average_umis_per_upi=summary_cell_calling["mean_umi_upia_cell"],
        fraction_reads_in_cells=summary_cell_calling["total_reads_cell"]
        / summary_all["reads"],
        median_antibodies_per_cell=summary_cell_calling["median_markers_cell"],
        total_antibodies_detected=summary_cell_calling["total_markers"],
        number_of_reads=summary_all["reads"],
        number_of_short_reads_skipped=summary_preqc["too_short_reads"],
        fraction_valid_pbs=summary_all["adapterqc"] / summary_all["reads"],
        fraction_valid_umis=summary_collapse["input"] / summary_all["reads"],
        sequencing_saturation=summary_all["duplication"],
        fraction_q30_bases_in_antibody_barcode=summary_amplicon["fraction_q30_barcode"],
        fraction_q30_bases_in_umi=summary_amplicon["fraction_q30_umi"],
        fraction_q30_bases_in_upia=summary_amplicon["fraction_q30_upia"],
        fraction_q30_bases_in_upib=summary_amplicon["fraction_q30_upib"],
        fraction_q30_bases_in_pbs1=summary_amplicon["fraction_q30_PBS1"],
        fraction_q30_bases_in_pbs2=summary_amplicon["fraction_q30_PBS2"],
        fraction_q30_bases_in_read=summary_amplicon["fraction_q30"],
        **antibodies_data_values,  # type: ignore
        **antibodies_data_fractions,  # type: ignore
    )

    data = collect_report_data(input_path, sample_id)

    # uses the default template
    builder = WebreportBuilder()

    report_path = str(Path(output_path) / f"{sample_id}_report.html")
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

    logger.debug("Creating web reports from %s", input_path)

    # TODO: Move file collection and workdir scanning
    #       logic of *_metrics functions to PixelatorWorkdir

    # collect and save metrics
    if panel is not None:
        panel_obj = load_antibody_panel(config, panel)
    else:
        panel_obj = None

    summary_amplicon = amplicon_metrics(input_path)
    summary_preqc = preqc_metrics(input_path)
    summary_adapterqc = adapterqc_metrics(input_path)
    summary_demux = demux_metrics(input_path)
    summary_collapse = collapse_metrics(input_path)
    summary_graph = graph_and_annotate_metrics(input_path, folder="graph")
    summary_annotate = graph_and_annotate_metrics(input_path, folder="annotate")
    summary_cell_calling = cell_calling_metrics(input_path)

    # a good sanity check is to make sure that the indexes
    # of all the summary tables are the same
    if not check_indexes(
        [
            summary_amplicon,
            summary_preqc,
            summary_adapterqc,
            summary_demux,
            summary_collapse,
            summary_graph,
            summary_annotate,
            summary_cell_calling,
        ]
    ):
        raise AssertionError(
            "Summary tables do not have the same number of samples:\n"
            f"amplicon: {summary_amplicon.index.to_list()}\n"
            f"preqc: {summary_preqc.index.to_list()}\n"
            f"adapterqc: {summary_adapterqc.index.to_list()}\n"
            f"demux: {summary_demux.index.to_list()}\n"
            f"collapse: {summary_collapse.index.to_list()}\n"
            f"graph: {summary_graph.index.to_list()}\n"
            f"annotate: {summary_annotate.index.to_list()}\n"
            f"cell_calling: {summary_cell_calling.index.to_list()}\n"
        )

    # add discarded column to summary_graph
    summary_graph["discarded"] = round(
        1 - (summary_graph["edges"] / summary_collapse["output_edges"]), 2
    )

    # add discarded column to summary_annotate
    summary_annotate["discarded"] = round(
        1 - (summary_annotate["edges"] / summary_graph["edges"]), 2
    )

    # add discarded column to summary_annotate
    summary_cell_calling["discarded"] = round(
        1 - (summary_annotate["edges"] / summary_collapse["output_edges"]), 2
    )

    # create a global summary table for all the main metrics in each stage
    summary_all = pd.DataFrame(
        {
            "reads": summary_preqc["total_reads"].to_numpy(),
            "preqc": summary_preqc["passed_filter_reads"].to_numpy(),
            "adapterqc": summary_adapterqc["output"].to_numpy(),
            "demux": summary_demux["output"].to_numpy(),
            "discarded": 0,
            "edges": summary_collapse["output_edges"].to_numpy(),
            "duplication": summary_collapse["duplication"].to_numpy(),
        },
        index=summary_preqc.index,
    )
    summary_all["discarded"] = round(
        1 - (summary_all["demux"] / summary_all["reads"]), 2
    )
    summary_all["avg_reads_umi"] = round(summary_all["demux"] / summary_all["edges"], 2)

    if metadata is not None:
        logger.debug("Parsing metadata from %s", metadata)
        metadata_df = pd.read_csv(metadata, sep=",", index_col=0)

        required_cols = {
            "sample_description",
            "panel_version",
            "panel_name",
        }
        if set(metadata_df.columns) != required_cols:
            raise AssertionError(
                f"The metadata must contain the following columns: {required_cols}"
            )

        if metadata_df.shape[0] == 0:
            raise AssertionError("The metadata must contain at least one sample")

        if metadata_df.shape[0] != len(set(metadata_df.index)):
            raise AssertionError("The metadata file contain unique sample ids")
    else:
        metadata_df = None

    # create the dynamic report (one per sample)
    for sample in summary_all.index.tolist():
        # parse metadata info if available
        if metadata is not None and metadata_df is not None:
            # we allow for a partial match of the sample id
            sample_metadata = metadata_df.filter(regex=sample, axis=0)
            if sample_metadata.shape[0] == 0:
                raise RuntimeError(f"Sample {sample} not found in metadata")
            if sample_metadata.shape[0] > 1:
                raise RuntimeError(f"Sample {sample} found multiple times in metadata")
            sample_metadata = sample_metadata.squeeze()
            sample_description = sample_metadata["sample_description"]
        else:
            sample_description = ""

        parameter_info = collect_parameter_info(input_path, sample)
        command_index, options_index = index_parameter_info(parameter_info)
        collapse_info = options_index.get("pixelator single-cell collapse")

        pixel_version = ""
        if collapse_info:
            design = collapse_info.get("--design")
            if design:
                pixel_version = design.value

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
            parameters=parameter_info,
        )

        create_dynamic_report(
            input_path=input_path,
            summary_all=summary_all.loc[sample, :],
            summary_amplicon=summary_amplicon.loc[sample, :],
            summary_preqc=summary_preqc.loc[sample, :],
            summary_demux=summary_demux.loc[sample, :],
            summary_collapse=summary_collapse.loc[sample, :],
            summary_annotate=summary_annotate.loc[sample, :],
            summary_cell_calling=summary_cell_calling.loc[sample, :],
            info=sample_info,
            output_path=output_path,
        )

    logger.debug("Finish creating web reports")
