"""
This module contains helper functions to collect necessary data for plots of
the dynamic webreport.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import json
import logging
import typing
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd
from anndata import AnnData

from pixelator.pixeldataset import SIZE_DEFINITION, PixelDataset
from pixelator.report.webreport.types import CommandInfo, CommandOption, WebreportData
from pixelator.report.workdir import PixelatorWorkdir

logger = logging.getLogger(__name__)


def components_umap_data(adata: AnnData) -> str:
    """
    Create a csv formatted string with the components umap data
    for he webreport.

    :param adata: an AnnData object with the umap data (obs)
    :return: a csv formatted string with umap data
    """
    empty = np.full(adata.n_obs, np.nan)

    if "X_umap" in adata.obsm:
        umap_df = pd.DataFrame(
            adata.obsm["X_umap"], columns=["umap1", "umap2"], index=adata.obs.index
        )
    else:
        umap_df = pd.DataFrame({"umap1": empty, "umap2": empty}, index=adata.obs.index)

    if "leiden" in adata.obs:
        umap_df["cluster"] = adata.obs["leiden"].to_numpy()
    else:
        umap_df["cluster"] = empty

    if "cluster_cell_class" in adata.obs:
        umap_df["cluster_cell_class"] = adata.obs["cluster_cell_class"].to_numpy()
    else:
        umap_df["cluster_cell_class"] = np.full(adata.n_obs, "unknown")

    umap_df["umis"] = adata.obs["edges"].to_numpy()

    return umap_df.to_csv(index=True, index_label="component")


def antibody_percentages_data(adata: AnnData) -> str:
    """
    Create the antibody percentages histogram data for the webreport.

    This function created a csv formatted string with the antibody name and the
    percentage of the antibody aggregated over all components.

    :param adata: an AnnData object with antibody counts percentages data
    :return: a csv formatted string with antibody percentages data
    """
    index = adata.var.index.set_names("antibody", inplace=False)
    df = pd.DataFrame({"percentage": adata.var["antibody_pct"]}, index=index)
    return df.to_csv()


def antibody_counts_data(adata: AnnData) -> str:
    """
    Create the antibody counts data for the webreport.

    :param adata: an AnnData object with the antibody counts data
    :return: a csv formatted string with the antibody counts data
    """
    return adata.to_df().to_csv(index=True, index_label="component")


def component_ranked_component_size_data(components_metrics: pd.DataFrame) -> str:
    """
    Create data for the `cell calling` and `component size distribution` plot
    in the webreport.

    This collects the component size and the number of antibodies per component.
    Components that pass the filters (is_filtered) are marked as selected.

    :param components_metrics: a pd.DataFrame with the components metrics
    :return: a csv formatted string with the plotting data
    """
    component_sizes = components_metrics[SIZE_DEFINITION].to_numpy()
    df = pd.DataFrame({"component_size": component_sizes})
    df["rank"] = df.rank(ascending=False, method="first")
    df["selected"] = components_metrics["is_filtered"].to_numpy()
    df["markers"] = components_metrics["antibodies"].to_numpy()
    df.sort_values(by="rank", inplace=True)
    return df.to_csv(index=True)


def collect_parameter_info(
    input_path: str, sample_name: Optional[str] = None
) -> List[CommandInfo]:
    """
    Collect all metainfo files from the workdir and generate parameter info.

    :param input_path: Path to the workdir
    :returns: A list of CommandInfo objects
    """
    # Function scope import to avoid circular dependencies
    from pixelator.cli import main_cli

    workdir = PixelatorWorkdir(input_path)
    param_files = workdir.metadata_files(sample_name)
    return generate_parameter_info(main_cli, param_files)


def _clean_commmand_path(click_context: click.Group, data: Dict[str, Any]) -> str:
    """Helper to remove pipeline CSV from the command list."""
    command_path = data["cli"]["command"].split(" ")

    # This is a hack to rewrite the weird command_path when using ctx.invoke
    # with the pipeline command
    rnd_pipeline_path = ("rnd", "pipeline", "CSV")
    pipeline_path = ("pipeline", "CSV")

    runs_from_pipeline_ctx = pipeline_path in zip(command_path, command_path[1:])
    runs_from_rnd_pipeline_ctx = rnd_pipeline_path in zip(
        command_path, command_path[1:], command_path[2:]
    )

    if runs_from_pipeline_ctx or runs_from_rnd_pipeline_ctx:
        path_to_remove = (
            rnd_pipeline_path if runs_from_rnd_pipeline_ctx else pipeline_path
        )
        command_path = [part for part in command_path if part not in path_to_remove]
        if runs_from_rnd_pipeline_ctx:
            command_path.insert(1, "single-cell")

    return " ".join(command_path)


def _find_click_command(
    click_context: click.Group, data: Dict[str, Any]
) -> click.Command:
    """
    Find the click command for a given parsed meta.json file.

    :param click_context: The click context of the main pixelator command
    :param data: The parsed meta.json file
    """
    command_path = data["cli"]["command"].split(" ")
    command_group: click.Group = click_context

    clean_command_path = _clean_commmand_path(click_context, data).split(" ")

    if len(command_path) <= 1:
        raise ValueError("Expected at least one subcommand")

    for subcommand in clean_command_path[1:-1]:
        if subcommand in command_group.commands:
            command_group = typing.cast(click.Group, command_group.commands[subcommand])
        else:
            raise ValueError(f"Unknown command {subcommand}")

    leaf_command = command_group.commands[clean_command_path[-1]]
    return leaf_command


def _process_meta_json_data(
    click_context: click.Group, data: Dict[str, Any]
) -> CommandInfo:
    """
    Process a single meta.json file and generate the parameter info.

    :params click_context: The click context of the main pixelator command
    :params file_data: The parsed meta.json file
    """

    leaf_command = _find_click_command(click_context, data)
    param_data: List[CommandOption] = []
    opt_lookup = {p.opts[0]: p for p in leaf_command.params}
    clean_command_name = _clean_commmand_path(click_context, data)

    for param_name, param_value in data["cli"]["options"].items():
        option_info = opt_lookup.get(param_name)

        if option_info is None:
            warnings.warn(
                f'Unknown parameter "{param_name}" for command: "{clean_command_name}"'
            )
            param_data.append(
                CommandOption(
                    name=param_name,
                    value=param_value,
                    default_value=None,
                    description=None,
                )
            )
            continue

        help_text = None
        if isinstance(option_info, click.Option):
            help_text = option_info.help

        param_data.append(
            CommandOption(
                name=param_name,
                value=param_value,
                default_value=option_info.default,
                description=help_text,
            )
        )

    command = CommandInfo(
        command=clean_command_name,
        options=param_data,
    )

    return command


def generate_parameter_info(
    click_context: click.Group, param_files: List[Path]
) -> List[CommandInfo]:
    """
    Combine and enrich commmand parameters for use in the webreport.

    :param click_context: The click context of the main pixelator command
    :param param_files: A list with paths of meta.json files
    :raises ValueError: If the command parsed from meta files is not found
        in the click context.
    :returns: A list of CommandInfo objects
    """
    data_flat: List[CommandInfo] = []
    order = list(click_context.commands["single-cell"].commands.keys())  # type: ignore

    for f in param_files:
        with open(f, "r") as fh:
            file_data = json.load(fh)

        command_info_flat = _process_meta_json_data(click_context, file_data)
        data_flat.append(command_info_flat)

    data_flat.sort(key=lambda x: order.index(x.command.split(" ")[-1]))
    return data_flat


def index_parameter_info(data: List[CommandInfo]):
    """
    Create two lookup tables for finding parameter info.

    :param data: the result from `generate_parameter_info`
    """
    command_index = {}
    comand_option_index: Dict[str, Dict[str, CommandOption]] = defaultdict(dict)

    for command_info in data:
        command_index[command_info.command] = command_info
        for option in command_info.options:
            comand_option_index[command_info.command][option.name] = option

    return command_index, comand_option_index


def collect_report_data(input_path: str, sample_id: str) -> WebreportData:
    """
    Collect the data needed to generate figures in the webreport.

    The `annotate` folder must be present in `input_path`.

    :param input_path: The path to the input folder
    :param sample_id: The sample id
    :raises NotADirectoryError: If the input folder is missing the annotate folder
    :raises FileNotFoundError: If the annotate folder is missing the datasets
    """
    logger.debug("Collecting web report data for %s in %s", sample_id, input_path)

    # check that the annotate folder is present
    source_path = Path(input_path) / "annotate"
    if not source_path.is_dir():
        raise NotADirectoryError(f"annotate folder missing in {source_path}")

    # parse filtered dataset
    dataset_file = source_path / f"{sample_id}.dataset.pxl"
    if not dataset_file.is_file():
        raise FileExistsError(f"dataset file {dataset_file} not found")

    dataset = PixelDataset.from_file(str(dataset_file))
    adata = dataset.adata
    component_data = components_umap_data(adata)
    antibody_percentages = antibody_percentages_data(adata)
    antibody_counts = antibody_counts_data(adata)

    # parse raw components metrics
    metrics_file = source_path / f"{sample_id}.raw_components_metrics.csv.gz"
    if not metrics_file.is_file():
        raise FileExistsError(f"components metrics file {metrics_file} not found")

    raw_components_metrics = pd.read_csv(str(metrics_file))
    ranked_component_size_data = component_ranked_component_size_data(
        raw_components_metrics
    )

    # build the report data
    data = WebreportData(
        component_data=component_data,
        ranked_component_size=ranked_component_size_data,
        antibodies_per_cell=None,
        sequencing_saturation=None,
        antibody_percentages=antibody_percentages,
        antibody_counts=antibody_counts,
    )

    logger.debug("Web report data collected for %s in %s", sample_id, input_path)
    return data
