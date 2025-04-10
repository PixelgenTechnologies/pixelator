"""Copyright © 2025 Pixelgen Technologies AB."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from pixelator.pna.demux.barcode_demuxer import DemuxRecordBatch
from pixelator.pna.demux.pipeline import (
    DemuxWriterProcess,
    DemuxWriterProcessMessages,
    PartsFilenamePolicy,
    mpctx,
)

batch1 = [
    [
        18,
        11,
        b"@\x8bv\xb3\xb1aE[\x03\xae\x01\x00\x05\\o\xeeV\xb8\xde\xe6\xa1\xe8\x00\x00\x80\xed\xdah1\x03\x00\x00",
    ],
    [
        0,
        0,
        b"6\xdc\xbam\xeb\xces;\xb7+\n\x00\xb5]\xa3\x05V\xafpm\xd7@\x0b\x00\xff\xff\xff\xff\xff\x1f\x00\x00",
    ],
    [
        49,
        47,
        b"[\xd1\x16\xed\xb0\xa2\xc3\x80\xc2\xad\r\x00u\xb1mm\x8d\xad.:\xcf\x9e\r\x00\x00jwsa\x17\x00\x00",
    ],
    [
        0,
        0,
        b"@\x8bv\xb3\xb1aE[\x03\xae\x01\x00\x05\\o\xeeV\xb8\xde\xe6\xa1\xe8\x00\x00\x80\xed\xdah1\x03\x00\x00",
    ],
    [
        34,
        15,
        b"p[\xa3\x9d\x81v[\xdb\x16\xa8\r\x00\xc0\xdca5\x86\xc20P\xb7\xb6\x01\x00\xadm\xd7\x18\x8c\x02\x00\x00",
    ],
    [
        0,
        0,
        b"\x80\xb1\xa2\xe86\xbb]\xdd\x02\x1d\n\x00\xab\xdb\xbam7\xa36\xdcn\xdb\x0c\x00\x86Qo\x1b0\x0c\x00\x00",
    ],
    [
        18,
        11,
        b"-Z\xb4\xee\x86\xc1F\x01w\xb5\x0b\x00\xc3\n\x0ch7{]\xdb\xadp\x07\x00\x83\x07l\xf0\xe0\x02\x00\x00",
    ],
    [
        0,
        0,
        b"-Z\xb4\xee\x86\xc1F\x01w\xb5\x0b\x00\xc3\n\x0ch7{]\xdb\xadp\x07\x00\xff\xff\xff\xff\xff\x1f\x00\x00",
    ],
    [
        29,
        47,
        b"[\xd1\x16\xed\xb0\xa2\xc3\x80\xc2\xad\r\x00u\xb1mm\x8d\xad.:\xcf\x9e\r\x00\x00jwsa\x17\x00\x00",
    ],
    [
        0,
        0,
        b".\\\x14\xde\xbaz+<\x14\xe8\x06\x00m\xbb\x16\x05\xda\xc2k[\xb7\xc6\x06\x00[a\xb8\x1d\x06\x00\x00\x00",
    ],
]
batch2 = [
    [
        56,
        18,
        b"-\xda\x16(P\xb4h\xdb\xad\x80\r\x00(\x86\xcd\x1b\xdc\xb6\xed\x8aa\x03\n\x00\xc0<\x0c@\x07\x14\x00\x00",
    ],
    [
        0,
        0,
        b"\x05\xd0\xcd\xf5j\xc0\x9d\xeb\xb5\x1d\n\x00sQ\xdb\xab\xe1\x19s\xed\xda6\x0c\x00]=\xd4\xde\xd0\r\x00\x00",
    ],
    [
        9,
        11,
        b"-\xda\x16(P\xb4h\xdb\xad\x80\r\x00(\x86\xcd\x1b\xdc\xb6\xed\x8aa\x03\n\x00\xc0<\x0c@\x07\x14\x00\x00",
    ],
    [
        0,
        0,
        b"-Z\xb4\xee\x86\xc1F\x01w\xb5\x0b\x00\xc3\n\x0ch7{]\xdb\xadp\x07\x00\xff\xff\xff\xff\xff\x1f\x00\x00",
    ],
    [
        9,
        11,
        b"\xe8\x80\x0e+\xda\x0e\x03\x8a\r\xc3\n\x00u\x01{]\xed\x1a\x98\x87a\x83\x01\x00\xf0\x00tE\x07\x00\x00\x00",
    ],
    [
        0,
        0,
        b"\xe8\x80\x0e+\xda\x0e\x03\x8a\r\xc3\n\x00u\x01{]\xed\x1a\x98\x87a\x83\x01\x00\xf0\x00tE\x07\x00\x00\x00",
    ],
    [
        34,
        15,
        b"\x18\x00\x17-\xea\xa2\xd8\xba\xb5m\r\x00u\x87\xd5v\rw\xf0j\x0f.\x0c\x00\xdeV\x03F\x07\x00\x00\x00",
    ],
    [
        0,
        0,
        b"X\x8b\xd9[\rl(\x00c[\x0b\x00F[\xcf\xf3\xba\xd9[\xe1\xa1\xee\x00\x00-\x06\xcf+\x06\x0f\x00\x00",
    ],
    [
        147,
        1,
        b"\x9b]\xa0\x9d\r`\x00\xd0\xb9m\x0b\x00\x80\xdbam[\xb7\xad\xb1\xc2\xe8\x06\x00\xd8:\xac\xed6\x14\x00\x00",
    ],
    [
        0,
        0,
        b"v]\x17(:wEQ\xa0@\x0b\x00\x1b\xdc\xb6p\x8b\xb6\xc5\xd0\x1a\x9e\x01\x00\x85\xd7y@g\x18\x00\x00",
    ],
]
batch3 = [
    [
        86,
        131,
        b"\xdd\x0c\x17n\xd7\xb56\xdc\xb9\x80\r\x00F]\x17\xee\x86b[\x8b\xae\x9b\x07\x00\x9e[\xd4u\x81\x01\x00\x00",
    ],
    [
        0,
        0,
        b"\xe8\x80\x0e+\xda\x0e\x03\x8a\r\xc3\n\x00u\x01{]\xed\x1a\x98\x87a\x83\x01\x00\xf0\x00tE\x07\x00\x00\x00",
    ],
    [
        11,
        11,
        b"\xe8\x80\x0e+\xda\x0e\x03\x8a\r\xc3\n\x00u\x01{]\xed\x1a\x98\x87a\x83\x01\x00\xf0\x00tE\x07\x00\x00\x00",
    ],
    [
        0,
        0,
        b"\x1b0\x14h[\xa0E\xbb\xa1\x03\n\x00\xae\x81m6`\x17\x80\xdd\x15\x80\x01\x00\x85mon\x07\x0c\x00\x00",
    ],
    [
        86,
        131,
        b"\x9b]\xa0\x9d\r`\x00\xd0\xb9m\x0b\x00\x80\xdbam[\xb7\xad\xb1\xc2\xe8\x06\x00\xd8:\xac\xed6\x14\x00\x00",
    ],
    [
        0,
        0,
        b"-\x8azE[\xafk\x81\r\x1b\x00\x00\x18\xbc\x0eug\xb4\xc0\xd6\x19F\x07\x00p=\x00\x036\x0c\x00\x00",
    ],
    [
        9,
        11,
        b"k;\x0f\x18\xbaa.\x80\xb9\xee\x0c\x00\xd8\xeav(\xd0\xd9\xc3\xda\x1a\x06\x00\x00\xff\xff\xff\xff\xff\x1f\x00\x00",
    ],
    [
        0,
        0,
        b"\xf3\xdc\x01\xc5\x8a\xb63j\x17+\n\x00\xf5\xd0\xd5mQ\x00\x9bW\x0c-\x06\x00s7\xdb\xf0\x00\x17\x00\x00",
    ],
    [
        49,
        80,
        b"\xf5\xb6b^\xbd\xa1+\x86\xc1\xed\x0c\x00\xad\x8b\xcdX\xb7akk\x03\xb6\x07\x00\xc3l\x0fE\x07\x14\x00\x00",
    ],
    [
        0,
        0,
        b"p\x0bt.\xb0\xbap\xdd\x0eE\x0b\x00\xc5\xda\x02h\xebu@Q\xb4@\x0b\x00sm\xa0\x00P\x00\x00\x00",
    ],
]


@pytest.fixture(scope="module")
def expected_df():
    marker_1 = []
    marker_2 = []
    molecule = []

    for b in [batch1, batch2, batch3]:
        for m1, m2, mol in b:
            marker_1.append(m1)
            marker_2.append(m2)
            molecule.append(mol)

    return pd.DataFrame.from_dict(
        {"marker_1": marker_1, "marker_2": marker_2, "molecule": molecule}
    )


def send_random_batches(queue):
    """Send a bunch of single row batches."""
    batch = DemuxRecordBatch()
    for b in [batch1, batch2, batch3]:
        for m1, m2, mol in b:
            batch.add_record(m1, m2, mol)
            batch_ipc = batch.serialize()
            queue.put([0, batch_ipc])
            batch.clear()

    queue.close()
    queue.join_thread()


def test_demux_writer_process(tmp_path, expected_df):
    output_dir = tmp_path
    policy = PartsFilenamePolicy("sample1")

    queue = mpctx.Queue(maxsize=2)
    writer_conn, writer_conn_child = mpctx.Pipe(duplex=True)

    writer_process = DemuxWriterProcess(
        output_directory=output_dir,
        filename_policy=policy,
        queue=queue,
        connection=writer_conn_child,
        schema=DemuxRecordBatch.schema(),
    )
    writer_process.start()

    writer_message = writer_conn.recv()
    assert writer_message == DemuxWriterProcessMessages.READY

    send_random_batches(queue)

    # Signal the writer process that we are done
    writer_conn.send(-1)
    writer_process.join()

    # Check that the files have been written
    assert (output_dir / "sample1.part_000.arrow").exists()
    with pa.ipc.open_file(output_dir / "sample1.part_000.arrow") as reader:
        df = reader.read_pandas()

    # Check that the content is as expected
    assert df.shape[0] == 30
    assert np.all(df == expected_df)
