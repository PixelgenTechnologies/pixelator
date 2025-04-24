"""Functions for the analysis of PixelDataset.

Copyright © 2022 Pixelgen Technologies AB.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pixelator.mpx.analysis.analysis_engine import PerComponentAnalysis, run_analysis
from pixelator.mpx.analysis.colocalization.types import TransformationTypes
from pixelator.mpx.analysis.polarization.types import PolarizationTransformationTypes
from pixelator.mpx.pixeldataset import (
    PixelDataset,
)
from pixelator.mpx.report.models import AnalysisSampleReport
from pixelator.mpx.report.models.analysis import (
    ColocalizationReport,
    PolarizationReport,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisParameters:
    """Analysis parameters."""

    compute_polarization: bool
    compute_colocalization: bool
    use_full_bipartite: bool
    polarization_transformation: PolarizationTransformationTypes
    polarization_n_permutations: int
    polarization_min_marker_count: int
    colocalization_transformation: TransformationTypes
    colocalization_neighbourhood_size: int
    colocalization_n_permutations: int
    colocalization_min_region_count: int
    colocalization_min_marker_count: int


def analyse_pixels(
    input: str,
    output: str,
    output_prefix: str,
    metrics_file: str,
    use_full_bipartite: bool,
    analysis_to_run: list[PerComponentAnalysis],
) -> None:
    """Run analysis functions on a PixelDataset.

    This function takes a pxl file that has been generated
    with `pixelator annotate`. The function then uses the `edge list` and
    the `AnnData` to compute the scores (polarization, co-abundance and
    co-localization) which are then added to the `PixelDataset` (depending
    on which scores are enabled).

    :param input: the path to the PixelDataset (zip)
    :param output: the path to the output file
    :param output_prefix: the prefix to prepend to the output file
    :param metrics_file: the path to a JSON file to write metrics
    :param use_full_bipartite: use the bipartite graph instead of the
                               one-node-projection (UPIA)
    :param analysis_to_run: a list of analysis functions (`PerComponentAnalysis` instances) to apply
                            to each component
    :param verbose: run if verbose mode when true
    :returns: None
    :rtype: None
    :raises AssertionError: the input arguments are not valid
    """
    logger.debug("Parsing PixelDataset from %s", input)

    # load the PixelDataset
    dataset = PixelDataset.from_file(input)

    names_of_analyses = {analysis.ANALYSIS_NAME for analysis in analysis_to_run}

    compute_polarization = "yes" if "polarization" in names_of_analyses else "no"
    compute_colocalization = "yes" if "colocalization" in names_of_analyses else "no"

    metrics = dict()
    metrics["polarization"] = "yes" if compute_polarization else "no"
    metrics["colocalization"] = "yes" if compute_colocalization else "no"

    dataset = run_analysis(
        pxl_dataset=dataset,
        analysis_to_run=analysis_to_run,
        use_full_bipartite=use_full_bipartite,
    )

    dataset.metadata["analysis"] = {
        "params": {
            analysis.ANALYSIS_NAME: analysis.parameters()
            for analysis in analysis_to_run
        }
    }
    # save dataset
    dataset.save(
        Path(output) / f"{output_prefix}.analysis.dataset.pxl", force_overwrite=True
    )

    polarization_report = PolarizationReport()
    colocalization_report = ColocalizationReport()

    report = AnalysisSampleReport(
        sample_id=output_prefix,
        polarization=polarization_report if compute_polarization else None,
        colocalization=colocalization_report if compute_colocalization else None,
    )
    report.write_json_file(metrics_file, indent=4)
