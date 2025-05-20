"""Code adapted from cutadapt.

Modifications Copyright © 2025 Pixelgen Technologies AB.
Under the same license terms as the original code.

Original copyright notice:
Copyright (c) 2010 Marcel Martin <marcel.martin@scilifelab.se> and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import enum
import io
import logging
import multiprocessing
import queue
import sys
import traceback
from collections import defaultdict
from multiprocessing import Queue
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol, Tuple, Type

import pyarrow as pa
import pyarrow.parquet as pq
from cutadapt.files import (
    FileFormat,
    InputFiles,
    InputPaths,
)
from cutadapt.report import Statistics
from cutadapt.runners import PipelineRunner
from cutadapt.utils import Progress

from pixelator.pna.demux.barcode_demuxer import BarcodeDemuxer, DemuxRecordBatch
from pixelator.pna.read_processing.runners import ReaderProcess, logger

logger = logging.getLogger()

mpctx = multiprocessing.get_context("spawn")


# See https://github.com/python/typeshed/issues/9860
if TYPE_CHECKING:
    mpctx_Process = multiprocessing.Process  # pragma: no cover
else:
    mpctx_Process = mpctx.Process


class DemuxPipeline:
    """A pipeline that processes reads and sends them to a writer queue for further processing."""

    def __init__(
        self, demuxer: BarcodeDemuxer, writer_queue: Optional[Queue] = None
    ) -> None:
        """Initialize a pipeline that processes reads and sends them to a writer queue for further processing.

        :param demuxer: The barcode demuxer to use
        :param writer_queue: The queue to send the processed reads to
        """
        self._demuxer = demuxer
        self._writer_queue = writer_queue

    def set_writer_queue(self, writer_queue: Queue) -> None:
        """Set the writer queue for the pipeline."""
        self._writer_queue = writer_queue

    def process_reads(
        self,
        infiles: InputFiles,
        progress: Optional[Progress] = None,
    ) -> Tuple[int, int, Optional[int]]:
        """Run the pipeline. Return statistics.

        :param infiles: Input chunks of data as a file-like object
        :param progress: A progress object to update
        """
        assert self._writer_queue is not None

        reader = infiles.open()
        n = 0  # no. of processed reads
        total_bp = 0
        _demuxer = self._demuxer

        for read in reader:
            n += 1
            if n % 10000 == 0 and progress is not None:
                progress.update(10000)

            total_bp += len(read)
            res = _demuxer(read)
            self._submit(res)

        if progress is not None:
            progress.update(n % 10000)

        infiles.close()
        return (n, total_bp, None)

    def flush(self):
        """Flush any buffered data from the pipeline."""
        for res in self._demuxer.flush():
            self._submit(res)

    def close(self):
        """Flush the remaining items to the queue and close the queue.

        Closing the queue and joining the tread managing it
        in the current active worker is needed to make sure
        all data is visible to consumers.
        """
        self.flush()
        self._writer_queue.close()
        self._writer_queue.join_thread()

    def _submit(self, res):
        if not res:
            return

        def _submit_single(r):
            submitted = False
            while not submitted:
                try:
                    self._writer_queue.put(r, timeout=5)
                    submitted = True
                except queue.Full:
                    pass

        if isinstance(res, list):
            for r in res:
                _submit_single(r)
        else:
            _submit_single(res)


class DemuxWriterProcessMessages(enum.IntEnum):
    """Enum for messages between different processes in the pipeline."""

    READY = 0
    DONE = 1
    EXCEPTION = 2
    SHUTDOWN = 3


class RecordBatchList:
    """A small helper class to store RecordBatches in a list and keep track of the total size."""

    def __init__(self):
        """Initialize a RecordBatchList instance."""
        self._batches = []
        self._total_num_rows = 0

    def append(self, batch: pa.RecordBatch):
        """Append a new RecordBatch to the list."""
        self._batches.append(batch)
        self._total_num_rows += batch.num_rows

    def batches(self) -> Iterator[pa.RecordBatch]:
        """Return an iterator over the RecordBatches."""
        return iter(self._batches)

    def clear(self):
        """Clear the list of RecordBatches."""
        self._batches.clear()
        self._total_num_rows = 0

    def num_rows(self):
        """Return the total size of RecordBatches in the list."""
        return self._total_num_rows


class DemuxFilenamePolicy(Protocol):
    """A policy to determine the filename of the output files.

    The filename is determined by the sample name and the group index.
    """

    def get_filename(self, group_index: int) -> str:
        """Return the filename for the given group index.

        :param group_index: The index of the group to write the data to
        """
        ...


class PartsFilenamePolicy(DemuxFilenamePolicy):
    """A filename policy for creating demux output files.

    Each group will be written to a separate file with the
    `part_???.demux.arrow` suffix.
    """

    def __init__(self, prefix: str):
        """Initialize the filename policy.

        :param prefix: The prefix to use for the output files.
        """
        super().__init__()
        self.prefix = prefix

    def get_filename(self, group_index: int) -> str:
        """Return the filename for the given group index."""
        return f"{self.prefix}.part_{group_index:03d}.parquet"


class IndependentMarkersFilenamePolicy(DemuxFilenamePolicy):
    """A filename policy for creating demux output files.

    The flat space of group_index is split into two parts,
    one for marker1 and one for marker2.

    Small marker groups can still be binned together, so the
    number of files is not equal to twice the number of markers
    in a panel.
    """

    def __init__(self, prefix: str, m1_map: dict[str, int], m2_map: dict[str, int]):
        """Initialize the filename policy.

        :param prefix: The prefix to use for the output files.
        """
        super().__init__()
        self.prefix = prefix

        # The set of group indices that are partitions for marker1
        self._m1_groups = set(m1_map.values())
        self._m1_min = min(self._m1_groups)
        # The set of group indices that are partitions for marker2
        self._m2_groups = set(m2_map.values())
        self._m2_min = min(self._m2_groups)

    def _rescaled_m1(self, idx: int):
        return idx - self._m1_min

    def _rescaled_m2(self, idx: int):
        return idx - self._m2_min

    def get_filename(self, group_index: int) -> str:
        """Return the filename for the given group index."""
        if group_index in self._m1_groups:
            idx = self._rescaled_m1(group_index)
            return f"{self.prefix}.m1.part_{idx:03d}.parquet"

        if group_index in self._m2_groups:
            idx = self._rescaled_m2(group_index)
            return f"{self.prefix}.m2.part_{idx:03d}.parquet"

        raise ValueError(f"Invalid group index {group_index}")


class DemuxWriterProcess(mpctx_Process):
    """A custom writer process for streaming data into multiple parquet files."""

    def __init__(
        self,
        output_directory: Path,
        filename_policy: DemuxFilenamePolicy,
        queue: multiprocessing.Queue,
        connection: Connection,
        schema: pa.schema,
    ):
        """Initialize a DemuxWriterProcess instance.

        :param output_directory: The directory to write the output arrow files to
        :param filename_prefix: The prefix to use for the output files
        :param queue: The queue to receive data from the worker processes
        :param connection: The connection to the parent process
        :param schema: The schema of the arrow Tables received from the worker nodes
        """
        super().__init__()

        # self._output_prefix = output_prefix
        #: The input connections to the individual worker processes
        self.output_directory = output_directory
        self.filename_policy = filename_policy
        self._queue = queue
        self._connection = connection
        self._write_buffers: dict[int, RecordBatchList] = defaultdict(RecordBatchList)
        self._batch_size = 10_000
        self._schema = schema

        self._batch_writers: dict[int, pq.ParquetWriter] = {}

        # Write uncompressed here
        # These IPC files will be compressed later to parquet after sorting.
        # Saving them uncompressed makes sure we can mmap the files in the sorting step
        # We save the write_option as a dict since `pa.ipc.IpcWriteOptions` is not picklable.
        self._write_options = {"use_threads": True}

    def open_writer(self, group_index: int):
        """Open a new writer for the given group index."""
        name = self.filename_policy.get_filename(group_index)
        output_file_path = str(self.output_directory / name)
        batch_writer = pq.ParquetWriter(
            output_file_path,
            self._schema,
            compression="zstd",
            compression_level=1,
        )
        self._batch_writers[group_index] = batch_writer
        return batch_writer

    def run(self):
        """Run the writer process."""
        # Give a ready signal to the parent process
        self._connection.send(DemuxWriterProcessMessages.READY)
        _queue = self._queue

        stop_signal = self._connection.poll(timeout=0)
        data = None

        try:
            while not stop_signal:
                try:
                    data = _queue.get(timeout=1)
                    group_index, group = data
                    self.buffer_batch_data(group_index, group)
                except Empty:
                    pass

                # Check for an out-of-band stop signal on the parent connection
                stop_signal = self._connection.poll(timeout=0)

            # Pump the queue empty, all remaining data is already present at this point
            while not _queue.empty():
                data = _queue.get()
                group_index, group = data
                self.buffer_batch_data(group_index, group)

            self.close()
            self._connection.send(DemuxWriterProcessMessages.DONE)
            self._connection.close()
        except Exception as e:
            self.close()
            self._connection.send(DemuxWriterProcessMessages.EXCEPTION)
            self._connection.send((e, traceback.format_exc()))

    def buffer_batch_data(self, group_index, buf: pa.Buffer):
        """Buffer the data from the worker processes in RecordBatches.

        If the buffer is large enough, write it to disk.

        :param group_index: The index of the group to write the data to
        :param buf: The serialized bytes of a RecordBatch received from a worker process
        """
        # Convert the raw buffer back into a RecordBatch
        group_batch = pa.ipc.read_record_batch(buf, self._schema)
        group_data = self._write_buffers[group_index]
        group_data.append(group_batch)

        # Check if we have enough data buffered to write a RecordBatch
        if group_data.num_rows() >= self._batch_size:
            self.write_batches(group_index, group_data)

    def write_batches(self, group_index: int, group_data: RecordBatchList) -> None:
        """Write batches of records to the output file for given `group_index`.

        :param group_index: The index of the group to write the data to
        :param group_data: The RecordBatchList containing the data to write
        """
        table = pa.Table.from_batches(group_data.batches())
        writer = self._batch_writers.get(group_index)

        if writer is None:
            writer = self.open_writer(group_index)

        writer.write_table(table)
        group_data.clear()

    def flush(self) -> None:
        """Flush any remaining data to the writers."""
        for group_index, group_data in self._write_buffers.items():
            if group_data.num_rows() > 0:
                self.write_batches(group_index, group_data)

    def close(self) -> None:
        """Close all writers."""
        self.flush()
        for writer in self._batch_writers.values():
            writer.close()


class DemuxWorkerProcess(mpctx_Process):
    """A custom process that runs the pipeline on chunks of input data.

    The worker process reads data from the read_pipe and processes it using the pipeline.
    The results are sent to the writer_queue for writing to different output files.

    To notify the reader process that it wants data, it puts its own identifier into the
    need_work_queue before attempting to read data from the read_pipe.
    """

    def __init__(
        self,
        id_: int,
        pipeline: DemuxPipeline,
        inpaths: InputPaths,
        read_pipe: Connection,
        parent_pipe: Connection,
        writer_queue: multiprocessing.Queue,
        need_work_queue: multiprocessing.Queue,
        statistics_class: Type[Statistics] = Statistics,
    ) -> None:
        """Initialize the worker process.

        :param id_: The index of the worker
        :param pipeline: The pipeline to run
        :param inpaths: The input paths to read data from
        :param read_pipe: The connection to the reader process
        :param parent_pipe: The connection to the parent process
        :param writer_queue: The queue to send the processed reads to
        :param need_work_queue: The queue to notify the reader process that work is needed
        """
        super().__init__()
        self._id = id_
        self._pipeline = pipeline
        self._n_input_files = len(inpaths.paths)
        self._interleaved_input = inpaths.interleaved
        self._read_pipe = read_pipe
        self._parent_pipe = parent_pipe
        self._writer_queue = writer_queue
        self._need_work_queue = need_work_queue
        self._statistics_class = statistics_class
        self._pipeline.set_writer_queue(self._writer_queue)

    def run(self):
        """Run the demux worker process."""
        try:
            stats = self._statistics_class()
            stats.paired = False

            while True:
                # Notify reader that we need data
                self._need_work_queue.put(self._id)
                chunk_index = self._read_pipe.recv()
                if chunk_index == -1:
                    # reader is done
                    break
                elif chunk_index == -2:
                    # An exception has occurred in the reader
                    e, tb_str = self._read_pipe.recv()
                    logger.error("%s", tb_str)
                    raise e

                # Receive blocks of data from the reader
                files = [
                    io.BytesIO(self._read_pipe.recv_bytes())
                    for _ in range(self._n_input_files)
                ]
                # Pass the data to the pipeline as a file like object
                infiles = InputFiles(*files, interleaved=self._interleaved_input)
                (n, bp1, bp2) = self._pipeline.process_reads(infiles)
                stats += self._statistics_class().collect(n, bp1, bp2, [], [])
                self._send_outfiles(chunk_index, n)

            self._pipeline.close()
            self._parent_pipe.send(-1)
            self._parent_pipe.send(stats)

        except Exception as e:
            self._pipeline.close()

            # Notify the parent process that an exception has occurred
            self._parent_pipe.send(-2)
            self._parent_pipe.send((e, traceback.format_exc()))

    def _send_outfiles(self, chunk_index: int, n_reads: int):
        self._parent_pipe.send(chunk_index)
        self._parent_pipe.send(n_reads)


class WorkerException(Exception):
    """Wrap an exception that occurred in a worker process.

    Capture the traceback string of the original exception.
    This is class is needed because traceback objects are not picklable.
    To communicate the traceback in a multiprocessing setting, we need to
    capture the traceback as a string.

    :ivar e: The original exception
    :ivar tb_str: The traceback string
    """

    def __init__(self, wrapped_exception, tb_str):
        """Initialize the WorkerException object.

        :param wrapped_exception: The original exception
        :param tb_str: The traceback string
        """
        self.e = wrapped_exception
        self.tb_str = tb_str

    def __str__(self):
        """Return a string representation of wrapped the exception."""
        return f"{self.e}\nwith traceback:\n{self.tb_str}"


class ParallelDemuxPipelineRunner(PipelineRunner):
    """Run a Demux pipeline and dispatch work to worker threads.

    - At construction, a reader process is spawned.
    - When run() is called, as many worker processes as requested are spawned.
    - The workers return a tuple of (file_index, data) to the main process.
    - The writer process buffers writes to the output files and writes them in the correct order.
    - write

    If a worker needs work, it puts its own index into a Queue() (_need_work_queue).
    The reader process listens on this queue and sends the raw data to the
    worker that has requested work. For sending the data from reader to worker,
    a Connection() is used. There is one such connection for each worker (self._pipes).

    For sending the processed data from the worker to the main process, there
    is a second set of connections, again one for each worker.

    When the reader is finished, it sends 'poison pills' to all workers.
    When a worker receives this, it sends a poison pill to the main process,
    followed by a Statistics object that contains statistics about all the reads
    processed by that worker.
    """

    def __init__(
        self,
        inpaths: InputPaths,
        n_workers: int,
        output_directory: Path,
        filename_policy: DemuxFilenamePolicy,
        buffer_size: Optional[int] = None,
    ):
        """Create a new ParallelDemuxPipelineRunner instance.

        Args:
            inpaths: The input paths to read data from.
            n_workers: The number of worker processes to spawn.
            output_directory: The directory to write the output files to.
            filename_policy: The policy to determine the filename of the output files.
            buffer_size: The size of the buffer to use for reading data.

        """
        self._n_workers = n_workers
        self._need_work_queue: multiprocessing.Queue = mpctx.Queue()
        self._buffer_size = 4 * 1024**2 if buffer_size is None else buffer_size
        self._inpaths = inpaths
        self._filename_policy = filename_policy

        if not output_directory.exists():
            raise NotADirectoryError(output_directory)

        output_directory.mkdir(parents=True, exist_ok=True)

        # the workers read from these connections
        in_connections = [mpctx.Pipe(duplex=False) for _ in range(self._n_workers)]
        self._worker_input_connection, connw = zip(*in_connections)

        # # Full-duplex pipe for communicating worker output with the writer process
        self._writer_queue = mpctx.Queue(maxsize=self._n_workers * 2)

        try:
            fileno = sys.stdin.fileno()
        except io.UnsupportedOperation:
            # This happens during tests: pytest sets sys.stdin to an object
            # that does not have a file descriptor.
            fileno = -1

        file_format_connection_r, file_format_connection_w = mpctx.Pipe(duplex=False)

        self._reader_process = ReaderProcess(
            *inpaths.paths,
            file_format_connection=file_format_connection_w,
            connections=connw,
            queue=self._need_work_queue,
            buffer_size=self._buffer_size,
            stdin_fd=fileno,
        )
        self._reader_process.daemon = True
        self._reader_process.start()

        self._writer_conn, writer_conn_child = mpctx.Pipe(duplex=True)

        self._writer_process = DemuxWriterProcess(
            output_directory=output_directory,
            filename_policy=self._filename_policy,
            queue=self._writer_queue,
            connection=writer_conn_child,
            schema=DemuxRecordBatch.schema(),
        )
        self._writer_process.daemon = True
        self._writer_process.start()

        self._input_file_format = self._try_receive(file_format_connection_r)

    def _start_workers(
        self, pipeline
    ) -> tuple[list[DemuxWorkerProcess], list[Connection]]:
        """Start the worker processes and return them and the connections to them."""
        workers = []
        out_connection_parent: list[Connection] = []

        for index in range(self._n_workers):
            conn_parent, conn_child = mpctx.Pipe(duplex=True)

            out_connection_parent.append(conn_parent)

            worker = DemuxWorkerProcess(
                index,
                pipeline,
                self._inpaths,
                read_pipe=self._worker_input_connection[index],
                parent_pipe=conn_child,
                writer_queue=self._writer_queue,
                need_work_queue=self._need_work_queue,
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)

        return workers, out_connection_parent

    def run(self, pipeline, progress) -> Statistics:
        """Run a pipeline on the pipeline runner.

        :param pipeline: The pipeline to run.
        :param progress: A progress object to update with the number of processed reads.
        :return: A statistics object.
        """
        workers, connections = self._start_workers(pipeline)

        # Wait for the writer process to signal it is ready to start receiving data
        writer_message = self._writer_conn.recv()
        if writer_message != DemuxWriterProcessMessages.READY:
            raise RuntimeError("Writer process did not signal ready")

        stats = Statistics()
        stats.paired = False

        while connections:
            # Regularly check if the writer process has a message
            writer_message = self._writer_conn.poll(timeout=0)

            if writer_message:
                status = self._writer_conn.recv()

                if status == DemuxWriterProcessMessages.DONE:
                    logger.info("Demux writer process finished")
                    self._writer_process.join()

                if status == DemuxWriterProcessMessages.EXCEPTION:
                    e, tb_str = self._writer_conn.recv()
                    logger.error("An exception occurred in the writer process")
                    logger.error("%s", tb_str)
                    for c in mpctx.active_children():
                        c.terminate()
                    raise e

            # Wait for any of the worker to be signal they are ready to receive work
            # We use a timeout to avoid blocking indefinitely so that we can check
            # for messages on the writer process as well
            try:
                ready_connections: list[Any] = multiprocessing.connection.wait(
                    connections, timeout=1
                )
            except TimeoutError:
                continue

            for connection in ready_connections:
                chunk_index: int = self._try_receive(connection)
                if chunk_index == -1:
                    # the worker is done, the second item send is the statistics
                    cur_stats = self._try_receive(connection)
                    stats += cur_stats
                    connections.remove(connection)
                    continue

                number_of_reads: int = self._try_receive(connection)
                progress.update(number_of_reads)

        for w in workers:
            w.join()

        self._reader_process.join()

        # Signal the writer process to stop
        self._writer_conn.send(-1)
        self._writer_process.join()
        progress.close()
        return stats

    @staticmethod
    def _try_receive_no_wait(connection):
        has_results = connection.poll(timeout=1)

        if not has_results:
            return None

        result = connection.recv()
        if result == -2:
            # An exception has occurred on the other end
            e, tb_str = connection.recv()
            # The other end does not send an actual traceback object because these are
            # not picklable, but a string representation.
            logger.debug("%s", tb_str)
            for child in multiprocessing.active_children():
                child.terminate()
            raise WorkerException(e, tb_str)

        return result

    @staticmethod
    def _try_receive(connection):
        """Try to receive data over `connection` and return it.

        If an exception was received, raise it.
        """
        result = connection.recv()
        if result == -2:
            # An exception has occurred on the other end
            e, tb_str = connection.recv()
            # The other end does not send an actual traceback object because these are
            # not picklable, but a string representation.
            logger.debug("%s", tb_str)
            for child in multiprocessing.active_children():
                child.terminate()
            raise WorkerException(e, tb_str)

        return result

    def close(self) -> None:
        """Close the pipeline runner."""
        pass

    def input_file_format(self) -> FileFormat:
        """Return the file format of the input files."""
        return self._input_file_format
