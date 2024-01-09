"""Functions for the analysis of PixelDataset.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from pixelator.analysis.colocalization import colocalization_scores
from pixelator.analysis.colocalization.types import TransformationTypes
from pixelator.analysis.polarization import polarization_scores
from pixelator.analysis.polarization.types import PolarizationNormalizationTypes
from pixelator.pixeldataset import (
    PixelDataset,
)
from pixelator.utils import np_encoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisParameters:
    """Analysis parameters."""

    compute_polarization: bool
    compute_colocalization: bool
    use_full_bipartite: bool
    polarization_normalization: PolarizationNormalizationTypes
    polarization_n_permutations: int
    colocalization_transformation: TransformationTypes
    colocalization_neighbourhood_size: int
    colocalization_n_permutations: int
    colocalization_min_region_count: int


def analyse_pixels(
    input: str,
    output: str,
    output_prefix: str,
    metrics_file: str,
    compute_polarization: bool,
    compute_colocalization: bool,
    use_full_bipartite: bool,
    polarization_normalization: PolarizationNormalizationTypes,
    polarization_n_permutations: int,
    colocalization_transformation: TransformationTypes,
    colocalization_neighbourhood_size: int,
    colocalization_n_permutations: int,
    colocalization_min_region_count: int,
    verbose: bool,
) -> None:
    """Calculate Moran's I statistics for a PixelDataset.

    This function takes a pxl file that has been generated
    with `pixelator annotate`. The function then uses the `edge list` and
    the `AnnData` to compute the scores (polarization, co-abundance and
    co-localization) which are then added to the `PixelDataset` (depending
    on which scores are enabled).

    :param input: the path to the PixelDataset (zip)
    :param output: the path to the output file
    :param output_prefix: the prefix to prepend to the output file
    :param metrics_file: the path to a JSON file to write metrics
    :param compute_polarization: compute polarization scores when True
    :param compute_colocalization: compute colocalization scores when True
    :param use_full_bipartite: use the bipartite graph instead of the
                               one-node-projection (UPIA)
    :param polarization_normalization: the method to use to normalize the
                                       antibody counts (raw or clr)
    :param polarization_n_permutations: Select number of permutations used to
                                        calculate empirical p-values of the
                                        polarization scores
    :param colocalization_transformation: Select a transformation method to use
                                          for the colocalization
    :param colocalization_neighbourhood_size: Set the size of the neighbourhood to
                                              consider when computing the colocalization
    :param colocalization_n_permutations: Select number of permutations used to
                                          calculate empirical p-values of the
                                          colocalization scores
    :param colocalization_min_region_count: The minimum size of the region (e.g. number
                                            of counts in the neighbourhood) required
                                            for it to be considered
    :param verbose: run if verbose mode when true
    :returns: None
    :rtype: None
    :raises AssertionError: the input arguments are not valid
    """
    logger.debug("Parsing PixelDataset from %s", input)

    # load the PixelDataset
    dataset = PixelDataset.from_file(input)
    edgelist = dataset.edgelist

    metrics = {}  # type: ignore
    metrics["polarization"] = "yes" if compute_polarization else "no"
    metrics["colocalization"] = "yes" if compute_colocalization else "no"

    # polarization scores
    if compute_polarization:
        # obtain polarization scores
        scores = polarization_scores(
            edgelist=edgelist,
            use_full_bipartite=use_full_bipartite,
            normalization=polarization_normalization,
            permutations=polarization_n_permutations,
        )
        dataset.polarization = scores

    # colocalization scores
    if compute_colocalization:
        # obtain colocalization scores
        scores = colocalization_scores(
            edgelist=edgelist,
            use_full_bipartite=use_full_bipartite,
            transformation=colocalization_transformation,
            neighbourhood_size=colocalization_neighbourhood_size,
            n_permutations=colocalization_n_permutations,
            min_region_count=colocalization_min_region_count,
        )
        dataset.colocalization = scores

    dataset.metadata["analysis"] = {
        "params": asdict(
            AnalysisParameters(
                compute_colocalization=compute_colocalization,
                compute_polarization=compute_polarization,
                use_full_bipartite=use_full_bipartite,
                polarization_normalization=polarization_normalization,
                polarization_n_permutations=polarization_n_permutations,
                colocalization_transformation=colocalization_transformation,
                colocalization_neighbourhood_size=colocalization_neighbourhood_size,
                colocalization_n_permutations=colocalization_n_permutations,
                colocalization_min_region_count=colocalization_min_region_count,
            )
        )
    }
    # save dataset
    dataset.save(str(Path(output) / f"{output_prefix}.dataset.pxl"))

    # save metrics (JSON)
    with open(metrics_file, "w") as outfile:
        json.dump(metrics, outfile, default=np_encoder)
