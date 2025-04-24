"""Copyright © 2025 Pixelgen Technologies AB."""

from cutadapt.info import ModificationInfo
from dnaio import SequenceRecord

from pixelator.pna.amplicon.process import QualityProfileStep
from pixelator.pna.config import pna_config


def test_quality_counting_per_region():
    assay = pna_config.get_assay("pna-2")
    step = QualityProfileStep(assay)

    s = SequenceRecord(
        name="amplicon1",
        sequence="TTGCCGCCGAACCATTGATCCTAATCAACAGCTATGGTCAAGTGACGCTGGGCATGTCAAACACTCATGTCGCTTTGCCGGTACGAGCTTCGCTTAGATGTCGGCAGCTATGGTCAACAAACGGATNCTTAACCACGATCCG",
        qualities="-CC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!!!!!!!!!!CCCCCCCCC;CCCCC;CCCCCCCCCCCCCCCCC-CCCCCC!!!!!!!!!!CCCCCCCCCCCC#CCCCCCCCCCCCCCC",
    )
    # TTGCCGCCGAACCATTGATCCTAATCAACAGCTATGGTCAAGTGACGCTGGGCATGTCAAACACTCATGTCGCTTTGCCGGTACGAGCTTCGCTTAGATGTCGGCAGCTATGGTCAACAAACGGATNCTTAACCACGATCCG
    # -CC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!!!!!!!!!!CCCCCCCCC;CCCCC;CCCCCCCCCCCCCCCCC-CCCCCC!!!!!!!!!!CCCCCCCCCCCC#CCCCCCCCCCCCCCC
    # [--------------------------][--------]CAAGTGACGCTGGGCATGTCAAACACTCATGTC[-------------]GCTTCGCTTAGATGTCGG[--------][--------------------------]
    #          umi-1                pid-1                  lbs-1                                 lbs-2           pid-2            umi-2

    info = ModificationInfo(s)
    step(s, info)

    s = step.get_statistics()

    assert s.total_bases("umi-1") == 28
    assert s.q30_bases("umi-1") == 26
    assert s.get_q30_fraction("umi-1") == 26 / 28

    assert s.total_bases("pid-1") == 10
    assert s.q30_bases("pid-1") == 10
    assert s.get_q30_fraction("pid-1") == 1.0

    assert s.total_bases("lbs-1") == 33
    assert s.sequences_bases("lbs-1") == 13
    assert s.q30_bases("lbs-1") == 13
    # We do not count unsequenced bases in the fraction here !
    assert s.get_q30_fraction("lbs-1") == 1.0

    assert s.total_bases("uei") == 15
    assert s.q30_bases("uei") == 13
    assert s.get_q30_fraction("uei") == 13 / 15

    assert s.total_bases("lbs-2") == 18
    assert s.q30_bases("lbs-2") == 17
    assert s.get_q30_fraction("lbs-2") == 17 / 18

    assert s.total_bases("umi-2") == 28
    assert s.sequences_bases("umi-2") == 27
    assert s.q30_bases("umi-2") == 27
    assert s.get_q30_fraction("umi-2") == 1.0
    assert s.get_q30_fraction_total_bases("umi-2") == 27 / 28

    assert s.total_bases("pid-2") == 10
    assert s.q30_bases("pid-2") == 0
    # check for zero division proof fractions
    assert s.get_q30_fraction("pid-2") == 0
