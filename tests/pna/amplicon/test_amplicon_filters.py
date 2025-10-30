"""Copyright © 2025 Pixelgen Technologies AB."""

from dnaio import SequenceRecord

from pixelator.pna.amplicon.filters import LBSDetectedInUMI, LowComplexityUMI
from pixelator.pna.config import pna_config


class TestLowComplexityUMI:
    @classmethod
    def setup_class(cls):
        assay = pna_config.get_assay("pna-2")
        cls.predicate = LowComplexityUMI(assay, proportion=0.80)

    def test_low_complexity_predicate(self):
        """Test that an assembled amplicon sequence passes the LowComplexityUMI filter."""
        #                      UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert not (
            self.predicate.test(SequenceRecord(name="pass", sequence=ampl1), None)
        )

    def test_low_complexity_umi1(self):
        """Test that an assembled amplicon sequence with low-complexity UMI-1 fails the filter."""

        #                      UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl2 = "CCCCCCCCCCCCCCCCCCCCCCCCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(
            SequenceRecord(name="fail-umi1", sequence=ampl2), None
        )

    def test_low_complexity_umi2(self):
        """Test that an assembled amplicon sequence with low-complexity UMI-2 fails the filter."""

        #                      UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl3 = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTTTTTTTTTTTTTTTTTCGTTTTT"
        assert self.predicate.test(
            SequenceRecord(name="fail-umi2", sequence=ampl3), None
        )


class TestLBSDetectedInUMI:
    @classmethod
    def setup_class(cls):
        assay = pna_config.get_assay("pna-2")
        cls.predicate = LBSDetectedInUMI(assay)

    def test_correct_sequence(self):
        #                      UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(SequenceRecord(name="pass", sequence=ampl1), None)

    def test_umi1_lbs1_match_end(self):
        #                      UMI1                                                                                                           UMI2
        #        ----------------------------|                                                                                    |----------------------------
        #        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCAGTTATCAAGTGACGCTGGGCATGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(
            SequenceRecord(name="umi_match_end", sequence=ampl1), None
        )

    def test_umi1_lbs1_match_start(self):
        #                     UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl = "GCTGGGCATGTCAAACACTCATGTCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(SequenceRecord(name="match", sequence=ampl), None)

    def test_umi1_lbs2_match_end(self):
        #                      UMI1                                                                                                           UMI2
        #        ----------------------------|                                                                                    |----------------------------
        #        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCGCTTCGCTTAGATTGGGCATGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(
            SequenceRecord(name="umi_match_end", sequence=ampl1), None
        )

    def test_umi1_lbs2_match_start(self):
        umi_filter = LBSDetectedInUMI(pna_config.get_assay("pna-2"))

        #                     UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl = "GCTTAGATGTCGGAACACTCATGTCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
        assert self.predicate.test(SequenceRecord(name="match", sequence=ampl), None)

    def test_umi2_lbs1_match_end(self):
        """Test that an assembled amplicon sequence with partial LBS matches is detected."""

        #                      UMI1                                                                                                           UMI2
        #        ----------------------------|                                                                                    |----------------------------
        #        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCAAGTGACGCTGG"
        assert self.predicate.test(
            SequenceRecord(name="umi_match_end", sequence=ampl1), None
        )

    def test_umi2_lbs1_match_start(self):
        #                     UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATTCAAACACTCATGTCACACCCAGGTTCCG"
        assert self.predicate.test(SequenceRecord(name="match", sequence=ampl), None)

    def test_umi2_lbs2_match_end(self):
        #                      UMI1                                                                                                           UMI2
        #        ----------------------------|                                                                                    |----------------------------
        #        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl1 = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCAGCTTCGCTTAG"
        assert self.predicate.test(
            SequenceRecord(name="umi_match_end", sequence=ampl1), None
        )

    def test_umi2_lbs2_match_start(self):
        """Test that an assembled amplicon sequence with partial LBS1 match is not flagged as an LBS containing read."""

        #                     UMI1                                                                                                           UMI2
        #       ----------------------------|                                                                                    |----------------------------
        #       NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
        ampl = "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATATTAGATGTCGGNAACCACCCAGGTTCCG"
        assert self.predicate.test(SequenceRecord(name="match", sequence=ampl), None)
