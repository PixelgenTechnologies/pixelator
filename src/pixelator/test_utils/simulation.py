"""Module containing utilities for simulating data.

Module containing utilities for simulating data. in testing, and in-silico experiments.
Copyright (c) 2023 Pixelgen Technologies AB.
"""

from typing import Generator, Iterable, Iterator, List, Optional

import numpy as np

from pixelator.config import Assay
from pixelator.config.config_instance import config


class ReadSimulator:
    """Simulate reads for tests and experiments.

    Simplistic read simulator that produces reads based on a assay configuration.
    It works by generating template molecule, that is then "sequenced" by sampling
    it a number of times. It allows for the addition of base substitutions errors
    with a given probability to simulate simple sequencing errors.
    """

    _ERRORS_DICT = {
        "A": ["C", "G", "T"],
        "C": ["A", "G", "T"],
        "G": ["C", "A", "T"],
        "T": ["C", "G", "A"],
    }

    def __init__(
        self,
        assay: Optional[Assay] = None,
        upia_pool_size: int = 1000,
        upib_pool_size: int = 1000,
        random_seed: int = 42,
        markers: List[str] = ["AAAAAAAA"],
    ) -> None:
        """Create a ReadSimulator instance.

        :param assay: an `Assay` to use will use D21 if None is passed,
                      defaults to None
        :param nbr_of_a_pixels: The number of A pixels to sample molecules from,
                                defaults to 1000
        :param nbr_of_b_pixels: The number of B pixels to sample molecules from,
                                defaults to 1000
        :param random_seed: Set the random seed, defaults to 42
        :param markers: A list of marker barcodes to use in generating the reads,
                        defaults to ["AAAAAAAA"]
        """
        if not assay:
            assay = config.assays["D21"]
        self.assay = assay
        self.rng = np.random.default_rng(random_seed)
        self.upia_pool = [self.random_seq(25) for _ in range(upia_pool_size)]
        self.upib_pool = [self.random_seq(25) for _ in range(upib_pool_size)]
        self.markers = markers

    def random_seq(self, length: int) -> str:
        """Generate a random DNA sequence of size length.

        :param length: length of the sequence to generate
        :return: A random DNA sequence
        :rtype: str
        """
        return "".join([self.rng.choice(["A", "T", "C", "G"]) for _ in range(length)])

    def build_molecule(self, nbr_of_molecules: int) -> Generator[str, None, None]:
        """Create an strings representing molecules, that can then be "sequenced".

        :param nbr_of_molecules: number of molecules to generate
        :yields: An iterator of DNA sequences
        :rtype: Generator[str, None, None]
        """

        def build_fragment():
            for region in self.assay.get_region_by_id("amplicon").regions:
                if region.region_id == "upi-b":
                    yield self.rng.choice(self.upib_pool)
                if region.region_id == "pbs-2":
                    yield region.sequence
                if region.region_id == "upi-a":
                    yield self.rng.choice(self.upia_pool)
                if region.region_id == "pbs-1":
                    yield region.sequence
                if region.region_id == "umi-b":
                    yield self.random_seq(region.max_len)
                if region.region_id == "bc":
                    yield self.rng.choice(self.markers)

        for _ in range(nbr_of_molecules):
            yield "".join(build_fragment())

    def sequence_molecule(
        self,
        molecules: Iterable[str],
        mean_nbr_of_reads_per_molecule: float,
        std_nbr_of_reads_per_molecule: float,
    ) -> Generator[str, None, None]:
        """Simulate sequencing of reads from a set of molecules.

        Generate "reads" of the input molecules, the number of times
        each molecule is read depends on `mean_nbr_of_reads_per_molecule`
        and `std_nbr_of_reads_per_molecule`.

        :param molecules: the underlying molecules to "sequence"
        :param mean_nbr_of_reads_per_molecule: the mean number of reads to generate per
                                               molecule
        :param std_nbr_of_reads_per_molecule: the standard deviation in the number of
                                              reads generated per molecule

        :yields: an iterator of "sequenced" reads
        :rtype: Generator[str, None, None]
        """
        for molecule in molecules:
            nbr_of_times_sequences = int(
                self.rng.normal(
                    mean_nbr_of_reads_per_molecule, std_nbr_of_reads_per_molecule
                )
            )
            for _ in range(nbr_of_times_sequences):
                yield molecule

    def add_sequencing_errors(
        self, reads: Iterable[str], error_prob_per_base: float
    ) -> Generator[str, None, None]:
        """Add sequencing errors to reads.

        Add base substitutions errors to `reads` with a probability
        given by `error_prop_per_base`.

        :param reads: reads to add errors to
        :param error_prob_per_base: probability of adding an error, range: [0,1)
        :yields: an iterator of DNA sequences with errors added to it
        :raises: Assertion error if `error_prob_per_base` is invalid
        :rtype: Generator[str, None, None]
        """
        if error_prob_per_base < 0 or error_prob_per_base > 1:
            raise AssertionError("`error_prob_per_base` must be between 0 and 1.")

        def add_errors(read):
            def data():
                for base in read:
                    if self.rng.random() < error_prob_per_base:
                        yield self.rng.choice(self._ERRORS_DICT[base])
                    else:
                        yield base

            return "".join(data())

        for read in reads:
            yield add_errors(read)

    def simulated_reads(
        self,
        n_molecules: int,
        mean_reads_per_molecule: float,
        std_reads_per_molecule: float,
        prob_of_seq_error: Optional[float] = 0,
    ) -> Iterator[str]:
        """Simulate a reads from a pool n molecules, adding sequencing errors.

        :param n_molecules: number of molecules to start from
        :param mean_reads_per_molecule: mean number of molecules to generate
                                        per molecule
        :param std_reads_per_molecule: the standard deviation of the number of
                                       molecules to generate
        :param prob_of_seq_error: probability of base substitutions errors per base
                                  sequenced. Default: 0.
        :return: An iterator of sequence reads
        :rtype: Iterator[str]
        """
        fragments = self.build_molecule(n_molecules)
        reads = self.sequence_molecule(
            fragments,
            mean_nbr_of_reads_per_molecule=mean_reads_per_molecule,
            std_nbr_of_reads_per_molecule=std_reads_per_molecule,
        )
        if prob_of_seq_error:
            return self.add_sequencing_errors(
                reads, error_prob_per_base=prob_of_seq_error
            )
        return reads
