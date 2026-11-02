"""Graph-collapse console script for pixelator (internal).

Copyright © 2026 Pixelgen Technologies AB.
"""

from pathlib import Path

import click

from pixelator.common.utils import (
    create_output_stage_dir,
    get_sample_name,
    log_step_start,
    sanity_check_inputs,
    timer,
    write_parameters_file,
)
from pixelator.pna import read
from pixelator.pna.cli.common import (
    design_option,
    logger,
    memory_option,
    output_option,
    threads_option,
)
from pixelator.pna.config import pna_config
from pixelator.pna.graph_collapse.collapse import run_graph_collapse
from pixelator.pna.graph_collapse.constants import (
    DEFAULT_BFS_DISTANCE,
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_MAX_DEGREE,
)
from pixelator.pna.graph_collapse.report import GraphCollapseSampleReport
from pixelator.pna.pixeldataset.io import PxlFile


@click.command(
    "graph-collapse",
    short_help=(
        "collapse likely UMI sequencing errors on low-degree nodes, per component, after the graph step"
    ),
    options_metavar="<options>",
)
@click.argument(
    "pxl_file",
    nargs=1,
    required=True,
    type=click.Path(exists=True),
    metavar="<PIXELFILE>",
)
@design_option
@click.option(
    "--bfs-distance",
    default=DEFAULT_BFS_DISTANCE,
    required=False,
    type=click.IntRange(min=2, max=None),
    show_default=True,
    help=(
        "Maximum number of hops from a searched node to look for a same-role collapse "
        "candidate."
    ),
)
@click.option(
    "--hamming-threshold",
    default=DEFAULT_HAMMING_THRESHOLD,
    required=False,
    type=click.IntRange(min=0, max=None),
    show_default=True,
    help=(
        "Maximum Hamming distance (inclusive) between a searched node's UMI sequence and a "
        "candidate's UMI sequence for the node to be collapsed into that candidate."
    ),
)
@click.option(
    "--report-collapsed-sequences",
    is_flag=True,
    default=False,
    help=(
        "Also write a TSV file listing, for every collapsed node, its component, original UMI "
        "sequence, and corrected UMI sequence (one per line)."
    ),
)
@click.option(
    "--max-degree",
    default=DEFAULT_MAX_DEGREE,
    required=False,
    type=click.IntRange(min=0, max=None),
    show_default=True,
    help=(
        "Search a node for a collapse target only when its degree is at most this "
        "value. The default, 1, investigates only leaves."
    ),
)
@output_option
@threads_option
@memory_option
@click.pass_context
@timer
def graph_collapse(
    ctx,
    pxl_file,
    design,
    bfs_distance,
    hamming_threshold,
    report_collapsed_sequences,
    max_degree,
    output,
    threads,
    memory,
):
    """Collapse likely UMI sequencing errors on low-degree nodes, per component.

    Since a UMI read error is unlikely to recur, an erroneous UMI typically shows up as a leaf
    node (degree 1) in its component's proximity graph. ``--max-degree`` chooses which nodes
    are searched; the default investigates only leaves. For each searched node, we search
    outward (breadth-first, up to ``--bfs-distance`` hops) for the nearest node of the same
    role (umi1 vs umi2) whose UMI is within ``--hamming-threshold`` of the node's own UMI,
    and rename the node to it. A chain of matches is followed to its end, and a rename is
    applied only when both UMIs have the same marker.
    """
    # Respect the global --cores when --threads is left at its default.
    threads = threads if threads and threads > 0 else ctx.obj["CORES"]

    input_files = [pxl_file]
    log_step_start(
        "graph-collapse",
        input_files=input_files,
        output=output,
        design=design,
        bfs_distance=bfs_distance,
        hamming_threshold=hamming_threshold,
        report_collapsed_sequences=report_collapsed_sequences,
        max_degree=max_degree,
        threads=threads,
        memory=memory,
    )

    sanity_check_inputs(input_files, allowed_extensions=("pxl",))

    sample_name = get_sample_name(pxl_file)
    graph_collapse_output = create_output_stage_dir(output, "graph-collapse")

    write_parameters_file(
        ctx,
        graph_collapse_output / f"{sample_name}.meta.json",
        command_path="pixelator single-cell-pna graph-collapse",
    )

    input_pxl_file = PxlFile(Path(pxl_file))
    output_path = graph_collapse_output / f"{sample_name}.graph-collapse.pxl"
    output_pxl_file_target = PxlFile.copy_pxl_file(input_pxl_file, output_path)

    assay = pna_config.get_assay(design)
    umi1_length = assay.get_region_by_id("umi-1").max_len
    umi2_length = assay.get_region_by_id("umi-2").max_len

    logger.info(f"Trying to collapse UMI errors on low-degree nodes in file {pxl_file}")

    input_dataset = read(input_pxl_file.path)
    reads_input = int(input_dataset.adata().obs["reads_in_component"].sum())
    umis_input = int(input_dataset.adata().obs["n_umi"].sum())

    result = run_graph_collapse(
        output_pxl_file_target,
        n_threads=threads,
        bfs_distance=bfs_distance,
        hamming_threshold=hamming_threshold,
        umi1_length=umi1_length,
        umi2_length=umi2_length,
        max_degree=max_degree,
        report_collapsed_sequences=report_collapsed_sequences,
        memory_limit=memory,
        temp_directory=ctx.obj.get("DUCKDB_TEMP_DIR"),
    )
    collapsed_dataset = read(output_pxl_file_target.path)

    reads_output = int(collapsed_dataset.adata().obs["reads_in_component"].sum())
    umis_output = int(collapsed_dataset.adata().obs["n_umi"].sum())

    collapsed_sequences_file = None
    if report_collapsed_sequences:
        collapsed_sequences_file = (
            graph_collapse_output
            / f"{sample_name}.graph-collapse.collapsed_sequences.tsv"
        )
        with open(collapsed_sequences_file, "w") as fh:
            fh.write("component\toriginal_sequence\tcorrected_sequence\n")
            for row in result.collapsed_sequences:
                fh.write(
                    f"{row['component']}\t{row['original_sequence']}\t"
                    f"{row['corrected_sequence']}\n"
                )

    metrics_file = graph_collapse_output / f"{sample_name}.report.json"
    report = GraphCollapseSampleReport(
        sample_id=sample_name,
        product_id="single-cell-pna",
        component_count=result.component_count,
        umis_input=umis_input,
        umis_output=umis_output,
        reads_input=reads_input,
        reads_output=reads_output,
        nodes_considered=result.nodes_considered,
        targets_found=result.targets_found,
        bfs_distance_distribution=result.bfs_distance_distribution,
        hamming_distance_distribution=result.hamming_distance_distribution,
        nodes_collapsed=result.nodes_collapsed,
        collapsed_sequences_file=(
            str(collapsed_sequences_file) if collapsed_sequences_file else None
        ),
    )
    report.write_json_file(metrics_file, indent=4)
