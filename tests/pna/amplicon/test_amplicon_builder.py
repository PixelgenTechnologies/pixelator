"""Copyright © 2025 Pixelgen Technologies AB."""

from cutadapt.info import ModificationInfo
from dnaio import SequenceRecord

from pixelator.pna.amplicon.build_amplicon import AmpliconBuilder
from pixelator.pna.config import pna_config


def test_amplicon_builder_no_mismatches():
    assay = pna_config.get_assay("pna-2")
    step = AmpliconBuilder(assay, mismatches=0.1)

    r1 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTG",
        qualities="CCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCGAAGCAACAAACTCCCCCCAGACATGA",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCC",
    )

    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)

    amplicon = step(r1, r2, info1, info2)

    # R1
    # TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTG

    # R2
    # CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCGAAGCAACAAACTCCCCCCAGACATGA
    # rev comp
    # TCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG

    # Template
    # NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    # TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTG--------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------TCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG

    assert (
        amplicon.sequence
        == "TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGGGGGAGTTTGTTGCTTCGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG"
    )
    assert (
        amplicon.qualities
        == "CCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!!!!!!!!!!CCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    )


def test_amplicon_builder_shift():
    assay = pna_config.get_assay("pna-2")
    step = AmpliconBuilder(assay, mismatches=0.2)

    r1 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="ACTGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAGGTG",
        qualities="CCCCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCAAAGCAACAAACTCCCCCCAGACATGA",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCC",
    )

    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)

    amplicon = step(r1, r2, info1, info2)
    assert amplicon

    # R1
    # ACTGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAGGTG

    # R2
    # CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCAAAGCAACAAACTCCCCCCAGACATGA
    # rev comp
    # TCATGTCTGGGGGGAGTTTGTTGCTTTGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG

    # Template
    # --NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    # ACTGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCAGGTG--------------------------------------------------------------------------------------------------
    #                                           X
    # ------------------------------------------------------------------TCATGTCTGGGGGGAGTTTGTTGCTTTGCTTAGATGTCGGTAAGACCATAGATTTAAGATTAAACCACCCAGGTTCCG


def test_amplicon_builder_large_shift():
    assay = pna_config.get_assay("pna-2")
    step = AmpliconBuilder(assay, mismatches=0.1)

    r1 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1101:68505:3745 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="CTGACAGGAAAAAATATATAGGGCCAGACCAAGTGACGCTGGGC",
        qualities="CC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1101:68505:3745 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGGAACCTGGGTCACTTTTGGTGACCGGATGGATGTGCCCGACATCTAAGCGAAGCCTCCCCTGCTACCGAGACATGA",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCC",
    )

    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)

    amplicon = step(r1, r2, info1, info2)
    assert amplicon is None


def test_uei_deletion():
    assay = pna_config.get_assay("pna-2")

    r1 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1101:29157:1511 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="GGCCCTCAGCTACGGAGAGTTCACTTAAGGCGAGAAATCAAGTG",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCC",
    )
    r2 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1101:29157:1511 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="TGATCCACCGGCTATGTTACGCTTGTGGGCCTCGCGCTCCGACATCTAAGCGAAGCCTCCTCCCCCGACATGAGTGTT",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC-CCCCCCCC",
    )
    # R2
    # TGATCCACCGGCTATGTTACGCTTGTGGGCCTCGCGCTCCGACATCTAAGCGAAGCCTCCTCCCCC-----GACATGAGTGTT
    # NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCCGACATCTAAGCGAAGCNNNNNNNNNNNNNNNGACATGAGTGTTTGACATGCCCAGCGTCACTTGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    # Rev comp template

    step = AmpliconBuilder(assay, mismatches=0.1)
    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)
    amplicon = step(r1, r2, info1, info2)

    assert amplicon

    assert amplicon.sequence == (
        "GGCCCTCAGCTACGGAGAGTTCACTTAAGGCGAGAAATCAAGTGACGCTGGGCATGTCAAACACTCATGTCGGGGGAGGAGNNNNNGCTTCGCTTAGATGTCGGAGCGCGAGGCCCACAAGCGTAACATAGCCGGTGGATCA"
    )
    #    NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    assert amplicon.qualities == (
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCC!!!!!!!!!!!!!!!CCCCCCCC-CCCCCCCCCCCCC!!!!!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    )


def test_bad_lbs2():
    assay = pna_config.get_assay("pna-2")

    r1 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:27434:1038 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="GTGAGACGGTGACGAAAACTACTTAGTACCAGGTTCCGCAAGTG",
        qualities=";CCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:27434:1038 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGAACCTGGTCTTTTGAACGCTCAGAGTGGTTAGATCCCGACATCTAAGCGATCTCTCTCTCTCTCTCTCTCTCTCTC",
        qualities="CC-CCCCCCCCCCC-CCCC;C;CCCCCCCCCCCC;CCCCCCCCCCCCC;CCCCC-CCCCCCCCCCCCCCCCCCCCCCC",
    )

    step = AmpliconBuilder(assay, mismatches=0.2)
    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)
    amplicon = step(r1, r2, info1, info2)

    assert amplicon is None
    assert step._custom_stats.failed_missing_upi2_umi2_reads == 1


def test_lbs2_insertion():
    assay = pna_config.get_assay("pna-2")

    r1 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1102:59037:42783 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="ATACACACACACGCATAAAGTTGCATAGCCCGTCATGACAAGTG",
        qualities="CC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    )
    r2 = SequenceRecord(
        name="VH00725:177:AAFHGNGM5:1:1102:59037:42783 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="TCATGACGGGTCTGTGACAAGGTAGGCAATTCACCGGTCCGACATCTAAGCGAAAGCCCCCCCAACTCGGCAGACATG",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCCCC;C",
    )

    step = AmpliconBuilder(assay, mismatches=0.2)
    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)
    amplicon = step(r1, r2, info1, info2)

    # Amplicon
    # NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    # CC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC!!!!!!!!!!!!!!!!!!!!!C;CCCCCCCCCCCCCCCCCCCCCCC-CC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
    # ATACACACACACGCATAAAGTTGCATAGCCCGTCATGACAAGTGACGCTGGGCATGTCAAACACTCATGTCTGCCGAGTTGGGGGGGCTT-CGCTTAGATGTCGGACCGGTGAATTGCCTACCTTGTCACAGACCCGTCATGA
    # ATACACACACACGCATAAAGTTGCATAGCCCGTCATGACAAGTG
    #                                                                  CATGTCTGCCGAGTTGGGGGGGCTTTCGCTTAGATGTCGGACCGGTGAATTGCCTACCTTGTCACAGACCCGTCATGA
    #                                                                  C;CCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

    assert amplicon
    assert (
        amplicon.sequence
        == "ATACACACACACGCATAAAGTTGCATAGCCCGTCATGACAAGTGACGCTGGGCATGTCAAACACTCATGTCTGCCGAGTTGGGGGGGCTTCGCTTAGATGTCGGACCGGTGAATTGCCTACCTTGTCACAGACCCGTCATGA"
    )


def test_amplicon_crappy_lbs1():
    assay = pna_config.get_assay("pna-2")
    step = AmpliconBuilder(assay, mismatches=0.1)

    r1 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCCCCCC",
        qualities="CCCCCC-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCGAAGCAACAAACTCCCCCCAGACATGA",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCC",
    )

    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)

    amplicon = step(r1, r2, info1, info2)

    assert amplicon is not None
    assert step._custom_stats.passed_missing_lbs1_anchor == 1


def test_amplicon_intersecting_reads():
    assay = pna_config.get_assay("pna-2")
    step = AmpliconBuilder(assay, mismatches=0.1)

    r1 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 1:N:0:AGGTCTTG+GATGAGGA",
        sequence="TGGCAAACGTCTGCAGTTATAAAGCTGACCAGGTTCCGCCCCCCTCTAAGCGAAGCAACAAACTCCCCCCAGACATGCGGAACCTGGCTTCGCTTAGATGTCGGCTATGGTCTTACCGACATCTAAGCGAAGCAACAAACTCCCCCCAG",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCC",
    )

    r2 = SequenceRecord(
        name="@VH00725:177:AAFHGNGM5:1:1101:65059:1057 2:N:0:AGGTCTTG+GATGAGGA",
        sequence="CGGAACCTGGGTGGTTTAATCTTAAATCTATGGTCTTACCGACATCTAAGCGAAGCAACAAACTCCCCCCAGACATGA",
        qualities="CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;CCCCCCCCCCCCCCCC",
    )

    info1 = ModificationInfo(r1)
    info2 = ModificationInfo(r2)

    amplicon = step(r1, r2, info1, info2)

    assert amplicon is not None
    assert step._custom_stats.passed_missing_lbs1_anchor == 1
