"""Demultiplex reads based on the PNA assay design.

Copyright © 2024 Pixelgen Technologies AB.
"""

import abc
import heapq
import math
import typing
from collections import Counter, defaultdict
from typing import Literal, Sequence, overload

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from dnaio import SequenceRecord

from pixelator.common.exceptions import PixelatorBaseException
from pixelator.pna.config import PNAAntibodyPanel, PNAAssay, get_position_in_parent


def create_barcode_group_to_batch_mapping(
    group_sizes: dict[tuple[str, str], int],
    reads_per_chunk: int = 50_000_000,
    max_chunks: int = 8,
) -> dict[tuple[str, str], int]:
    """Create groups of (PID1, PID2) pairs.

    The input mapping is a dictionary of (PID1, PID2) pairs to the number of
    reads that have been found to contain that pair of PID barcodes.

    The total number of reads in a supergroup can be less or more than the target_chunk_size.

    :param group_sizes: the mapping of (PID1, PID2) pairs to the number of reads
    :param reads_per_chunk: the target number of reads per chunk
    :param max_chunks: the maximum number of groups
    """
    # We can initialize the group mapping already with a default batch index of 0

    marker_counts: typing.Counter[tuple[str, str]] = Counter()

    for keys, v in group_sizes.items():
        marker_counts[keys] += v

    target_chunks = int(
        min(
            math.ceil(sum(marker_counts.values()) / reads_per_chunk),
            max_chunks,
        )
    )

    marker_labels = partition_greedy(list(marker_counts.values()), target_chunks)
    marker_group_mapping = {
        marker: int(group) for marker, group in zip(marker_counts.keys(), marker_labels)
    }

    return marker_group_mapping


def partition_greedy(items: Sequence[int], n: int) -> npt.NDArray[np.int32]:
    """Greedy number partitioning of a set of items with a given weight into n subsets.

    Each subset has an approximately equal sum of weights.

    :param counts: a dictionary of items and their weights
    :param n: the number of subsets
    """
    # Initialize a priority queue with n empty subsets
    subset_sums = [(0, i) for i in range(n)]
    heapq.heapify(subset_sums)

    labels = np.zeros(len(items), dtype=np.int32)

    for idx, count in enumerate(sorted(items, reverse=True)):
        # Pop the subset with the smallest sum
        current_sum, subset_index = heapq.heappop(subset_sums)
        # Assign the item to this subset
        labels[idx] = subset_index
        # Push the updated subset back into the priority queue
        heapq.heappush(subset_sums, (current_sum + count, subset_index))

    return labels


K = typing.TypeVar("K", bound=tuple[str, str] | str)


def independent_marker_groups_mapping(
    group_sizes: dict[tuple[str, str], int],
    reads_per_chunk: int = 50_000_000,
    max_chunks: int = 8,
) -> tuple[dict[str, int], dict[str, int]]:
    """Combine grouped reads by markers into approximately equal sized chunks.

    This is the classical partitioning problem.
    We use the simple greedy number partitioning algorithm here.

    The number of groups is determined by the `target_chunk_reads`.
    The number of groups will be capped by `max_chunk_count` should the calculated number of groups
    exceed this value.

    Params:
        group_sizes: the mapping of (marker1, marker2) pairs to the number of reads
        target_chunk_count: the number of chunks to partition the markers into

    Returns:
        A tuple with a dict for marker1 and marker2. The dict map the marker to the group index.

    """
    marker1_counts: typing.Counter[str] = Counter()
    marker2_counts: typing.Counter[str] = Counter()

    for (m1, m2), v in group_sizes.items():
        marker1_counts[m1] += v
        marker2_counts[m2] += v

    target_m1_chunks = int(
        min(
            math.ceil(sum(marker1_counts.values()) / reads_per_chunk),
            max_chunks,
        )
    )
    target_m2_chunks = int(
        min(
            math.ceil(sum(marker2_counts.values()) / reads_per_chunk),
            max_chunks,
        )
    )

    marker1_labels = partition_greedy(list(marker1_counts.values()), target_m1_chunks)
    marker1_group_mapping = {
        marker: int(group)
        for marker, group in zip(marker1_counts.keys(), marker1_labels)
    }

    offset = np.max(marker1_labels) + 1
    marker2_labels = partition_greedy(list(marker2_counts.values()), target_m2_chunks)
    marker2_group_mapping = {
        marker: int(offset + group)
        for marker, group in zip(marker2_counts.keys(), marker2_labels)
    }

    return marker1_group_mapping, marker2_group_mapping


class DemuxRecordBatch:
    """A batch of demux records.

    Batches are used to collect records before they are serialized and sent to the writer process.
    A batch uses a pre-allocated fixed length numpy array per field to store the records.

    :params batch_size: the maximum number of records in the batch
    """

    # TODO: Make this modular by using the assay design
    #   the packed array lengths for umi's and uei are hard-coded now
    _record_batch_schema = pa.schema(
        [
            pa.field("marker_1", pa.uint16()),
            pa.field("marker_2", pa.uint16()),
            pa.field("molecule", pa.binary(32)),
        ]
    )

    def __init__(self, capacity=10_000):
        """Initialize the DemuxRecordBatch object.

        :param capacity: the maximum number of records in the batch
        """
        self._batch_size = capacity
        self._size = 0
        self.marker1 = np.zeros(self._batch_size, dtype=np.uint16)
        self.marker2 = np.zeros(self._batch_size, dtype=np.uint16)
        self.molecule = np.zeros(self._batch_size, dtype="V32")

    @classmethod
    def schema(cls) -> pa.Schema:
        """Return the schema of the record batch returned by `to_arrow`."""
        return cls._record_batch_schema

    def add_record(self, marker1: int, marker2: int, molecule: bytes):
        """Add a new record to the batch.

        :param marker1: the first marker index
        :param marker2: the second marker index
        :param molecule: the molecule embedding
        """
        _size = self._size

        if _size < self._batch_size:
            self.marker1[_size] = marker1
            self.marker2[_size] = marker2
            self.molecule[_size] = molecule
            self._size += 1
        else:
            raise IndexError("Batch is full")

    def __len__(self) -> int:
        """Return the current number of records in the batch."""
        return self._size

    def capacity(self) -> int:
        """Return the maximum number of records in the batch."""
        return self._batch_size

    def clear(self):
        """Clear the batch.

        Note that the memory is not released, only the size is reset to 0.
        """
        self._size = 0

    def _verify_m1_markers(self, allowed_ids: Sequence[int]) -> bool:
        """Check if all markers are in the allowed set."""
        mask = np.isin(self.marker1[: self._size], allowed_ids)
        return bool(np.all(mask))

    def _verify_m2_markers(self, allowed_ids: Sequence[int]) -> bool:
        """Check if all markers are in the allowed set."""
        mask = np.isin(self.marker2[: self._size], allowed_ids)
        return bool(np.all(mask))

    def to_arrow(self) -> pa.RecordBatch:
        """Convert a number of cached records to a pyarrow RecordBatch."""
        _size = self._size
        # Passing the size option to pa.array does not seem to work for some reason.
        # So we are using slices instead.
        return pa.RecordBatch.from_arrays(
            [
                pa.array(
                    self.marker1[:_size],
                    type=self._record_batch_schema.field("marker_1").type,
                ),
                pa.array(
                    self.marker2[:_size],
                    type=self._record_batch_schema.field("marker_2").type,
                ),
                pa.array(
                    self.molecule[:_size],
                    type=self._record_batch_schema.field("molecule").type,
                ),
            ],
            schema=self._record_batch_schema,
        )

    def serialize(self) -> pa.Buffer:
        """Serialize the record batch as Arrow IPC message."""
        return self.to_arrow().serialize()


class BarcodeDemuxingError(PixelatorBaseException):
    """An error occurred during barcode demultiplexing."""

    pass


_ENCODING_LUT = np.zeros((256, 3), dtype=np.uint8)

# Numpy array representation is in most to least significant bit order

_ENCODING_LUT[ord("A")] = np.array([0, 1, 1], dtype=np.uint8)  # b'110'
_ENCODING_LUT[ord("C")] = np.array([1, 0, 1], dtype=np.uint8)  # b'101'
_ENCODING_LUT[ord("T")] = np.array([1, 1, 0], dtype=np.uint8)  # b'011'
_ENCODING_LUT[ord("G")] = np.array([0, 0, 0], dtype=np.uint8)  # b'000'


_ENCODING_LUT_2BIT = np.zeros((256, 2), dtype=np.uint8)

_ENCODING_LUT_2BIT[ord("A")] = np.array([0, 0], dtype=np.uint8)  # b'00'
_ENCODING_LUT_2BIT[ord("C")] = np.array([1, 0], dtype=np.uint8)  # b'01'
_ENCODING_LUT_2BIT[ord("G")] = np.array([0, 1], dtype=np.uint8)  # b'10'
_ENCODING_LUT_2BIT[ord("T")] = np.array([1, 1], dtype=np.uint8)  # b'11'

_DECODING_LUT = np.zeros((256), dtype=np.uint8)
_DECODING_LUT[6] = ord(b"A")
_DECODING_LUT[5] = ord(b"C")
_DECODING_LUT[3] = ord(b"T")
_DECODING_LUT[0] = ord(b"G")
_DECODING_LUT[7] = ord(b"N")

_RECODING_3BIT_TO_2BIT_LUT = np.zeros((8, 2), dtype=np.uint8)

_RECODING_3BIT_TO_2BIT_LUT[6] = np.array([0, 0], dtype=np.uint8)  # A -> b'00'
_RECODING_3BIT_TO_2BIT_LUT[5] = np.array([1, 0], dtype=np.uint8)  # C -> b'01'
_RECODING_3BIT_TO_2BIT_LUT[0] = np.array([0, 1], dtype=np.uint8)  # G -> b'01'
_RECODING_3BIT_TO_2BIT_LUT[3] = np.array([1, 1], dtype=np.uint8)  # T -> b'11'


class PNAEmbedding:
    """A class for constructing a compact binary embedding of a PNA amplicon reads.

    The embedding assumes a maximum length of 32 nucleotides
    for each UMI and 15 nucleotides for the UEI.
    """

    def __init__(self, assay: PNAAssay):
        """Initialize the PNAEmbedding object.

        :param assay: the assay design
        """
        self.assay = assay

        self._umi1_len = assay.get_region_by_id("umi-1").max_len
        self._umi2_len = assay.get_region_by_id("umi-2").max_len
        self._uei_len = assay.get_region_by_id("uei").max_len

        if self._umi1_len > 32:
            raise RuntimeError("UMI1s longer than 32 nucleotides are not supported")
        if self._umi2_len > 32:
            raise RuntimeError("UMI2s longer than 32 nucleotides are not supported")
        if self._uei_len > 21:
            raise RuntimeError("UEIs longer than 21 nucleotides are not supported")

        self._umi1_slice = slice(0, self._umi1_len * 3)
        self._umi2_slice = slice(96, 96 + self._umi2_len * 3)
        self._uei_slice = slice(192, 192 + self._uei_len * 3)

    def get_umi1_bytes(self, embedded: bytes):
        """Return the bytes of the UMI1 from the molecule embedding."""
        return embedded[0:12]

    def get_umi2_bytes(self, embedded: bytes):
        """Return the bytes of the UMI2 from the molecule embedding."""
        return embedded[12:24]

    def get_uei_bytes(self, embedded: bytes):
        """Return the bytes of the UEI from the molecule embedding."""
        return embedded[24:32]

    def _encode_umi1(self, umi1: bytes, output):
        assert len(umi1) == self._umi1_len

        umi1_arr = np.frombuffer(umi1, dtype=np.uint8, count=len(umi1))
        umi1_vec_slice = output[self._umi1_slice].reshape((self._umi1_len, 3))
        np.take(_ENCODING_LUT, umi1_arr, axis=0, out=umi1_vec_slice)

    def _encode_umi2(self, umi2: bytes, output):
        assert len(umi2) == self._umi2_len

        umi1_arr = np.frombuffer(umi2, dtype=np.uint8, count=len(umi2))
        umi1_vec_slice = output[self._umi2_slice].reshape((self._umi1_len, 3))
        np.take(_ENCODING_LUT, umi1_arr, axis=0, out=umi1_vec_slice)

    def _encode_uei(self, uei: bytes, output):
        assert len(uei) == self._uei_len

        uei_arr = np.frombuffer(uei, dtype=np.uint8, count=len(uei))

        # Write all ones if the UEI is not present
        # Otherwise encode the UEI into the last 8 bytes
        # 15 nucleotides * 3 bits = 45 bits
        # padded to 64 bits = 8 bytes
        if np.any(uei_arr == ord("N")):
            output[self._uei_slice] = 1
        else:
            uei_vec_slice = output[self._uei_slice].reshape((15, 3))
            np.take(_ENCODING_LUT, uei_arr, axis=0, out=uei_vec_slice)

    def encode(self, umi1: bytes, umi2: bytes, uei: bytes):
        """Pack the unique molecule identifiers into a single 256-bit vector.

        Each nucleotide is packed into 3 bits, so 8 nucleotides per 24-bit
        Padding bits are added to make the total length 256 bits and to separate
        the regions on byte boundaries.

        256 bits is a common length for SIMD instructions (eg. 1 AVX-2 register or 2 Arm NEON registers)
        and thus commonly used for efficient similarity search. (eg. in FAISS).
        """
        vec = np.zeros(256, dtype=np.uint8)

        self._encode_umi1(umi1, vec)
        self._encode_umi2(umi2, vec)
        self._encode_uei(uei, vec)

        # Pack the binary vector into bytes
        res = np.packbits(vec, bitorder="little")

        return res

    @overload
    def decode(
        self, bitvector: np.ndarray, skip_uei: Literal[True]
    ) -> tuple[bytes, bytes]: ...

    @overload
    def decode(
        self, bitvector: np.ndarray, skip_uei: Literal[False]
    ) -> tuple[bytes, bytes, bytes]: ...

    def decode(
        self, bitvector: npt.NDArray[np.uint8] | bytes, skip_uei: bool = False
    ) -> tuple[bytes, bytes] | tuple[bytes, bytes, bytes]:
        """Unpack the 84-bit UMI1 bitvector from a byte array.

        :param bitvector: the 256-bit vector to unpack
        :param skip_uei: whether to skip unpacking the UEI
        """
        if isinstance(bitvector, bytes):
            array_view = np.frombuffer(bitvector, dtype=np.uint8, count=len(bitvector))
        else:
            array_view = bitvector

        vec = np.unpackbits(array_view, bitorder="little")

        umi1_vec_slice = vec[self._umi1_slice].reshape((self._umi1_len, 3))
        packed_umi1 = np.packbits(umi1_vec_slice, axis=1, bitorder="little")
        umi1_array = np.take(_DECODING_LUT, packed_umi1, axis=0)
        umi1 = umi1_array.tobytes()

        umi2_vec_slice = vec[self._umi2_slice].reshape((self._umi2_len, 3))
        packed_umi2 = np.packbits(umi2_vec_slice, axis=1, bitorder="little")
        umi2_array = np.take(_DECODING_LUT, packed_umi2, axis=0)
        umi2 = umi2_array.tobytes()

        if skip_uei:
            return umi1, umi2

        uei_vec_slice = vec[self._uei_slice].reshape((self._uei_len, 3))
        packed_uei = np.packbits(uei_vec_slice, axis=1, bitorder="little")
        uei_array = np.take(_DECODING_LUT, packed_uei, axis=0)
        uei = uei_array.tobytes()

        return umi1, umi2, uei

    def encode_umi(self, umi: bytes):
        """Encode a single umi into a 3-bit per nucleotide encoded 128-bit vector.

        Each nucleotide is packed into 3 bits, so 8 nucleotides per 24-bit
        The output is padded to 128 bits since that is a common SIMD vector length
        and thus commonly used for efficient similarity search. (eg. in FAISS).

        :returns: a 16-byte/128-bits array of packed nucleotides
        :raises ValueError: if the UMI is longer than 32 nucleotides
        """
        if len(umi) > 40:
            raise ValueError("UMI cannot be longer than 40 nucleotides")

        vec = np.zeros(128, dtype=np.uint8)
        self._encode_umi1(umi, vec)

        # Pack the binary vector into bytes
        res = np.packbits(vec, bitorder="little")

        return res.tobytes()

    def decode_umi(self, umi_bytes: npt.NDArray[np.uint8] | bytes) -> bytes:
        """Decode a 3-bit encoded 128-bit umi vector into a nucleotide sequence.

        :param umi_bytes: the 128-bit umi vector
        :returns: the decoded nucleotide sequence
        :raises ValueError: if the input vector is not 128 bits long
        """
        if isinstance(umi_bytes, bytes):
            bytes_view = np.frombuffer(umi_bytes, dtype=np.uint8, count=len(umi_bytes))
        else:
            bytes_view = umi_bytes

        vec = np.unpackbits(bytes_view, bitorder="little")

        umi_vec_slice = vec[self._umi1_slice].reshape((self._umi1_len, 3))
        packed_umi = np.packbits(umi_vec_slice, axis=1, bitorder="little")
        umi_array = np.take(_DECODING_LUT, packed_umi, axis=0)
        umi = umi_array.tobytes()
        return umi

    def compress_umi_embedding(
        self, bitvector: npt.NDArray[np.uint8] | bytes
    ) -> np.uint64:
        """Compress the 3-bit per nucleotide into a more classical 2-bit embedding.

        The nucleotides encoded in this space are not equidistant anymore, but it is more compact.

        :param bitvector: the 128-bit umi vector
        :returns: the new 2-bit embedding for given input
        :raises ValueError: if the input vector is not 128 bits long
        """
        return self._compress_3bit_embedding(bitvector, 12, 28)

    def compress_uei_embedding(
        self, bitvector: npt.NDArray[np.uint8] | bytes
    ) -> np.uint64:
        """Compress the 3-bit per nucleotide embedded UEI into a 2-bit embedding."""
        return self._compress_3bit_embedding(bitvector, 8, 15)

    def _compress_3bit_embedding(
        self, bitvector: npt.NDArray[np.uint8] | bytes, n: int, decoded_len: int
    ) -> np.uint64:
        """Compress the 3-bit per nucleotide into a more classical 2-bit embedding.

        The nucleotides encoded in this space are not equidistant anymore, but it is more compact.

        Args:
            bitvector: A 3bit encoded byte array
            n: The length of encoded bytes without padding.
            decoded_len: The length of the uncompressed sequence

        Returns:
            the new 2-bit embedding for given input

        Raises:
            ValueError: if the input vector is not 128 bits long

        """
        array_view: npt.NDArray[np.uint8]

        if isinstance(bitvector, bytes):
            input_len = min(len(bitvector), n)
            array_view = np.zeros(n, dtype=np.uint8)
            array_view[:input_len] = np.frombuffer(
                bitvector, dtype=np.uint8, count=input_len
            )
        else:
            array_view = bitvector

        bitslice = slice(0, decoded_len * 3)

        vec = np.unpackbits(array_view, bitorder="little")
        vec_slice = vec[bitslice].reshape((decoded_len, 3))
        numeric_nucl_array = np.packbits(vec_slice, axis=1, bitorder="little")

        out = np.take(_RECODING_3BIT_TO_2BIT_LUT, numeric_nucl_array, axis=0)
        out = np.squeeze(out, 1)

        padded_2bit = np.zeros((8,), dtype=np.uint8)
        tmp = np.packbits(out, bitorder="little")
        padded_2bit[0 : len(tmp)] = tmp
        out_int = np.frombuffer(padded_2bit, dtype=np.uint64, count=1)[0]
        return out_int


class BarcodeDemuxer(abc.ABC):
    """An abstract base class for BarcodeDemuxer steps."""

    def __init__(self, assay: PNAAssay, panel: PNAAntibodyPanel):
        """Initialize a BarcodeDemuxer object.

        :param assay: the assay design
        :param panel: the antibody panel
        """
        self.assay = assay
        self.panel = panel
        self._pid1_slice = slice(*get_position_in_parent(assay, "pid-1"))
        self._pid2_slice = slice(*get_position_in_parent(assay, "pid-2"))
        self._umi1_slice = slice(*get_position_in_parent(assay, "umi-1"))
        self._umi2_slice = slice(*get_position_in_parent(assay, "umi-2"))
        self._uei_slice = slice(*get_position_in_parent(assay, "uei"))

        self._records_handled = 0
        self._records_written = 0
        self._output_groups_buffer: dict[int, DemuxRecordBatch] = defaultdict(
            DemuxRecordBatch
        )

        # 4 bits per nucleotide into bytes
        self._packed_umi1_len = (
            (self._umi1_slice.stop - self._umi1_slice.start) * 4 // 8
        )
        self._packed_umi2_len = (
            (self._umi2_slice.stop - self._umi2_slice.start) * 4 // 8
        )
        self._packed_uei_len = (self._uei_slice.stop - self._uei_slice.start) * 4 // 8

        self._embedding = PNAEmbedding(assay)

    def _serialize_group(
        self, group_id: int, group: DemuxRecordBatch
    ) -> tuple[int, bytes]:
        """Serialize and send a group to the writer process."""
        serialized_batch = group.serialize()
        self._records_written += len(group)
        group.clear()
        return (group_id, serialized_batch)

    def flush(self):
        """Return all remaining groups."""
        for group_id, group in self._output_groups_buffer.items():
            if len(group) > 0:
                yield self._serialize_group(group_id, group)
                group.clear()

        self._output_groups_buffer.clear()
        self._records_handled = 0
        self._records_written = 0

    @abc.abstractmethod
    def __call__(
        self, read: SequenceRecord
    ) -> list[tuple[int, bytes]] | tuple[int, bytes] | None:
        """Find the nearest antibody for a given barcode.

        :param read: the input read to process
        :returns: A tuple or list of tuples with a barcode_group_id and a serialized batch of records
            The serialized batch is encoded in the pyarrow RecordBatch ipc format.
        """
        raise NotImplementedError


class IndependentBarcodeDemuxer(BarcodeDemuxer):
    """Find the nearest antibody for a given barcode.

    The marker1 and marker2 barcode are grouped independently.
    Each read will be present twice in the output, once for each barcode.
    """

    def __init__(
        self,
        assay: PNAAssay,
        panel: PNAAntibodyPanel,
        marker1_groups: dict[str, int],
        marker2_groups: dict[str, int],
    ):
        """Initialize the BarcodeIdentifier object.

        :param assay: the assay design
        :param panel: the antibody panel
        :param marker1_groups: the mapping of marker1 to group id
        :param marker2_groups: the mapping of marker2 to group id
        """
        super().__init__(assay, panel)
        self.marker1_groups = marker1_groups
        self.marker2_groups = marker2_groups

    def __call__(self, read: SequenceRecord) -> list[tuple[int, bytes]] | None:
        """Find the nearest antibody for a given barcode.

        :param read: the read to process
        :return: the read with the antibody information added
        """
        if read.comment is None:
            raise BarcodeDemuxingError("No comment found in read")

        pid_comment = read.comment.split(" ")[-1]
        parts = pid_comment.split(":")
        if len(parts) != 2:
            raise BarcodeDemuxingError(f"Invalid PID comment: {pid_comment}")

        pid1, pid2 = pid_comment.split(":")
        # pid1, pid2 = self.record_group_mapping.get(read.id, (None, None))
        if pid1 is None or pid2 is None:
            return None

        read_bytes = read.sequence.encode("ascii")
        umi1 = read_bytes[self._umi1_slice]
        umi2 = read_bytes[self._umi2_slice]
        uei = read_bytes[self._uei_slice]

        vector = self._embedding.encode(umi1, umi2, uei)

        marker1_group_id = self.marker1_groups.get(pid1)
        marker2_group_id = self.marker2_groups.get(pid2)

        if marker1_group_id is None or marker2_group_id is None:
            raise BarcodeDemuxingError(
                f"One of the markers was not found in panel: {pid1}, {pid2}"
            )

        pid1_idx = self.panel.markers.index(pid1)
        pid2_idx = self.panel.markers.index(pid2)

        marker1_group = self._output_groups_buffer[marker1_group_id]
        marker1_group.add_record(pid1_idx, pid2_idx, vector)

        marker2_group = self._output_groups_buffer[marker2_group_id]
        marker2_group.add_record(pid1_idx, pid2_idx, vector)

        results = []
        if len(marker1_group) == marker1_group.capacity():
            results.append(self._serialize_group(marker1_group_id, marker1_group))

        if len(marker2_group) == marker2_group.capacity():
            results.append(self._serialize_group(marker2_group_id, marker2_group))

        return results or None


class PairedBarcodeDemuxer(BarcodeDemuxer):
    """Demux reads according to (PID1, PID2) groups.

    Each (PID1, PID2) pair is assigned to group id.

    """

    def __init__(
        self,
        assay: PNAAssay,
        panel: PNAAntibodyPanel,
        supergroups: dict[tuple[str, str], int],
    ):
        """Initialize the PairedBarcodeDemuxer object.

        :param assay: the assay design
        :param panel: the antibody panel
        :param supergroups: the mapping of (PID1, PID2) pairs to supergroup ids
        """
        super().__init__(assay, panel)
        self._supergroups = supergroups
        self._buffer_len = 10_000

    def __call__(self, read: SequenceRecord) -> tuple[int, bytes] | None:
        """Find the nearest antibody for a given barcode.

        :param read: the read to process
        :return: the read with the antibody information added
        """
        if read.comment is None:
            raise BarcodeDemuxingError("No comment found in read")

        pid_comment = read.comment.split(" ")[-1]
        parts = pid_comment.split(":")
        if len(parts) != 2:
            raise BarcodeDemuxingError(f"Invalid PID comment: {pid_comment}")

        pid1, pid2 = parts
        if pid1 is None or pid2 is None:
            return None

        read_bytes = read.sequence.encode("ascii")
        umi1 = read_bytes[self._umi1_slice]
        umi2 = read_bytes[self._umi2_slice]
        uei = read_bytes[self._uei_slice]
        vector = self._embedding.encode(umi1, umi2, uei)
        group_id = self._supergroups[(pid1, pid2)]

        if group_id is None:
            return None

        pid1_idx = self.panel.markers.index(pid1)
        pid2_idx = self.panel.markers.index(pid2)

        group = self._output_groups_buffer[group_id]
        group.add_record(pid1_idx, pid2_idx, vector)
        self._records_handled += 1

        if len(group) == group.capacity():
            return self._serialize_group(group_id, group)

        return None
