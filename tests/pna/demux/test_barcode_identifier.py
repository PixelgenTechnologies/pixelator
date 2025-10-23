"""Copyright © 2025 Pixelgen Technologies AB.

Test the BarcodeIdentifier class.

BarcodeIdentifier is part of the first pass of the demultiplexing pipeline.
It is responsible for identifying the antibody pair associated with a given barcode.

Copyright (c) Pixelgen Technologies AB
"""

import tempfile

from cutadapt.files import ProxyRecordWriter
from dnaio import SequenceRecord

from pixelator.pna.config import pna_config
from pixelator.pna.demux.barcode_identifier import BarcodeIdentifier


def test_barcode_identifier_exact_match():
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")

    barcodes_id = BarcodeIdentifier(assay=assay, panel=panel, mismatches=1)
    #                                             PID1                                                                    PID2
    #                                       [--------]                                                                  [--------]
    # template "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    amplicon = "AGGCCGCAGTGACGAGGTGTCTCTTCTGATAGGTTAGGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGATCGTTCCATAAGCTTCGCTTAGATGTCGGATATGACGGTGGGCCTTTACTTAGGCTTGAAAAAATCA"

    # ATAGGTTAGG = CD16
    # ATATGACGGT = CD11c

    s = SequenceRecord(
        name="@VH00725:211:AAG225HM5:1:1102:48755:1000 1:N:0:CCGTACAG+AGAACTGT",
        sequence=amplicon,
        qualities="III9IIIIIIIIIIIIIIIIII9IIIIIIIIIIIIIIIIIIIII!!!!!!!!!!!!!!!!!!!!IIIIIIII9IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    )
    original_name = s.name
    r = barcodes_id(s)

    # Check that the name has been updated
    assert r.name == original_name + " CD16:CD11c"

    # Check that the sequence has not been modified
    assert r.sequence == amplicon

    # Check that the statistics have been updated
    assert barcodes_id._stats.passed == 1
    assert barcodes_id._stats.pid_pair_counter[("CD16", "CD11c")] == 1


def test_barcode_identifier_one_mismatch():
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")

    barcodes_id = BarcodeIdentifier(assay=assay, panel=panel, mismatches=1)
    #                                             PID1                                                                    PID2
    #                                       [--------]                                                                  [--------]
    # template "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    amplicon = "AGGCCGCAGTGACGAGGTGTCTCTTCTGATAAGTTAGGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGATCGTTCCATAAGCTTCGCTTAGATGTCGGATATGACGGTGGGCCTTTACTTAGGCTTGAAAAAATCA"

    # ATAAGTTAGG = CD16 (1 mismatch)
    # ATATGACGGT = CD11c

    s = SequenceRecord(
        name="@VH00725:211:AAG225HM5:1:1102:48755:1000 1:N:0:CCGTACAG+AGAACTGT",
        sequence=amplicon,
        qualities="III9IIIIIIIIIIIIIIIIII9IIIIIIIIIIIIIIIIIIIII!!!!!!!!!!!!!!!!!!!!IIIIIIII9IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    )
    original_name = s.name
    r = barcodes_id(s)

    # Check that the name has been updated
    assert r.name == original_name + " CD16:CD11c"

    # Check that the sequence has not been modified
    assert r.sequence == amplicon

    # Check that the statistics have been updated
    assert barcodes_id._stats.passed == 1
    assert barcodes_id._stats.pid_pair_counter[("CD16", "CD11c")] == 1


def test_barcode_identifier_failed():
    assay = pna_config.get_assay("pna-2")
    panel = pna_config.get_panel("proxiome-immuno-155")


    failed_writer = ProxyRecordWriter([tempfile.NamedTemporaryFile().name], fileformat="fastq")

    barcodes_id = BarcodeIdentifier(
        assay=assay,
        panel=panel,
        mismatches=1,
        writer=failed_writer,
    )

    #                                             PID1                                                                    PID2
    #                                       [--------]                                                                  [--------]
    # template "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAAGTGACGCTGGGCATGTCAAACACTCATGTCNNNNNNNNNNNNNNNGCTTCGCTTAGATGTCGGNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
    amplicon = "AGGCCGCAGTGACGAGGTGTCTCTTCTGATCCATTAGGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGATCGTTCCATAAGCTTCGCTTAGATGTCGGATCTGCCGGTGGGCCTTTACTTAGGCTTGAAAAAATCA"

    s = SequenceRecord(
        name="VH00725:211:AAG225HM5:1:1102:48755:1000 1:N:0:CCGTACAG+AGAACTGT",
        sequence=amplicon,
        qualities="III9IIIIIIIIIIIIIIIIII9IIIIIIIIIIIIIIIIIIIII!!!!!!!!!!!!!!!!!!!!IIIIIIII9IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",
    )
    original_name = s.name
    r = barcodes_id(s)

    # Check that the statistics have been updated
    assert barcodes_id._stats.failed == 1
    assert barcodes_id._stats.missing_pid1_pid2 == 1

    chunks = failed_writer.drain()

    # Check that the failed read has been written and None:None is added to the header
    assert len(chunks) == 1
    assert chunks[0] == (
        "@VH00725:211:AAG225HM5:1:1102:48755:1000 1:N:0:CCGTACAG+AGAACTGT None:None\n"
        "AGGCCGCAGTGACGAGGTGTCTCTTCTGATCCATTAGGCAAGTGACGCTGGGCATGTCAAACACTCATGTCTGGATCGTTCCATAAGCTTCGCTTAGATGTCGGATCTGCCGGTGGGCCTTTACTTAGGCTTGAAAAAATCA\n"
        "+\n"
        "III9IIIIIIIIIIIIIIIIII9IIIIIIIIIIIIIIIIIIIII!!!!!!!!!!!!!!!!!!!!IIIIIIII9IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    ).encode("ascii")
