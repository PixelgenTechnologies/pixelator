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

import io
import logging
import multiprocessing
import os
import sys
import traceback
import typing
from abc import ABC, abstractmethod
from contextlib import ExitStack
from multiprocessing.connection import Connection
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import dnaio
from cutadapt.files import (
    FileFormat,
    InputFiles,
    InputPaths,
    OutputFiles,
    ProxyWriter,
    detect_file_format,
    xopen_rb_raise_limit,
)
from cutadapt.pipeline import Pipeline
from cutadapt.report import Statistics
from cutadapt.utils import Progress

logger = logging.getLogger()

mpctx = multiprocessing.get_context("spawn")

# See https://github.com/python/typeshed/issues/9860
if TYPE_CHECKING:
    mpctx_Process = multiprocessing.Process  # pragma: no cover
else:
    mpctx_Process = mpctx.Process


class ReaderProcess(mpctx_Process):
    """Read chunks of FASTA or FASTQ data (single-end or paired) and send them to a worker.

    The reader repeatedly

    - reads a chunk from the file(s)
    - reads a worker index from the Queue
    - sends the chunk to connections[index]

    and finally sends the stop token -1 ("poison pills") to all connections.
    """

    def __init__(
        self,
        *paths: str,
        file_format_connection: Connection,
        connections: Sequence[Connection],
        queue: multiprocessing.Queue,
        buffer_size: int,
        stdin_fd,
    ):
        """Initialize a ReaderProcess.

        :param params paths: path to input files
        :param connections: a list of Connection objects, one for each worker.
        :param queue: a Queue of worker indices. A worker writes its own index into this
        :param queue to notify the reader that it is ready to receive more data.
        :param buffer_size:
        :param stdin_fd:

        .. note::
            This expects the paths to the input files as strings because these can be pickled
            while file-like objects such as BufferedReader cannot. When using multiprocessing with
            the "spawn" method, which is the default method on macOS, function arguments must be
            picklable.
        """
        super().__init__()
        if len(paths) > 2:
            raise ValueError("Reading from more than two files currently not supported")
        if not paths:
            raise ValueError("Must provide at least one file")
        self._paths = paths
        self._file_format_connection = file_format_connection
        self.connections = connections
        self.queue = queue
        self.buffer_size = buffer_size
        self.stdin_fd = stdin_fd

    def run(self):
        """Read chunks of data from the input files and send them to workers."""
        if self.stdin_fd != -1:
            sys.stdin.close()
            sys.stdin = os.fdopen(self.stdin_fd)
        try:
            with ExitStack() as stack:
                try:
                    files = [
                        stack.enter_context(xopen_rb_raise_limit(path))
                        for path in self._paths
                    ]
                    file_format = detect_file_format(files[0])
                except Exception as e:
                    self._file_format_connection.send(-2)
                    self._file_format_connection.send((e, traceback.format_exc()))
                    raise
                self._file_format_connection.send(file_format)
                for index, chunks in enumerate(self._read_chunks(*files)):
                    self.send_to_worker(index, *chunks)
            self.shutdown()
        except Exception as e:
            # TODO better send this to a common "something went wrong" Queue
            # This code is rarely executed because there is little that can go wrong
            # splitting up the input into chunks. FASTQ/FASTA parsing problems
            # are caught within the workers.
            for connection in self.connections:
                connection.send(-2)
                connection.send((e, traceback.format_exc()))

    def _read_chunks(self, *files) -> Iterator[Tuple[memoryview, ...]]:
        if len(files) == 1:
            for chunk in dnaio.read_chunks(files[0], self.buffer_size):
                yield (chunk,)
        elif len(files) == 2:
            for chunks in dnaio.read_paired_chunks(
                files[0], files[1], self.buffer_size
            ):
                yield chunks
        else:
            raise NotImplementedError

    def send_to_worker(self, chunk_index, chunk1, chunk2=None):
        """Send a chunk of data to a worker.

        :param chunk_index: The index of the chunk.
        :param chunk1: The first chunk of data.
        :param chunk2: The second chunk of data (if paired-end data).
        """
        worker_index = self.queue.get()
        connection = self.connections[worker_index]
        connection.send(chunk_index)
        connection.send_bytes(chunk1)
        if chunk2 is not None:
            connection.send_bytes(chunk2)

    def shutdown(self):
        """Notify all workers to shut down."""
        # Send `-1` poison pills to all workers
        for _ in range(len(self.connections)):
            worker_index = self.queue.get()
            self.connections[worker_index].send(-1)


class WorkerProcess(mpctx_Process):
    """A custom multiprocessing Proces for running a pipeline on chunks of data.

    The worker repeatedly reads chunks of data from the read_pipe, runs the pipeline on it
    and sends the processed chunks to the write_pipe.

    To notify the reader process that it wants data, it puts its own identifier into the
    need_work_queue before attempting to read data from the read_pipe.
    """

    def __init__(
        self,
        id_: int,
        pipeline: Pipeline,
        inpaths: InputPaths,
        proxy_files: List[ProxyWriter],
        read_pipe: Connection,
        write_pipe: Connection,
        need_work_queue: multiprocessing.Queue,
        statistics_class: Type[Statistics] = Statistics,
    ):
        """Initialize a WorkerProcess."""
        super().__init__()
        self._id = id_
        self._pipeline = pipeline
        self._n_input_files = len(inpaths.paths)
        self._interleaved_input = inpaths.interleaved
        self._read_pipe = read_pipe
        self._write_pipe = write_pipe
        self._need_work_queue = need_work_queue
        self._proxy_files = proxy_files
        self._statistics_class = statistics_class

    def run(self):
        """Read chunks of data from the reader, process them and send them to the writer."""
        try:
            stats = self._statistics_class()

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

            stats_instance = self._statistics_class()
            stats += stats_instance.collect(
                0,
                0,
                0 if self._pipeline.paired else None,
                self._pipeline._modifiers,
                self._pipeline._steps,
                set_paired_to_none=self._pipeline.paired is None,
            )
            self._write_pipe.send(-1)
            self._write_pipe.send(stats)

        except Exception as e:
            self._write_pipe.send(-2)
            self._write_pipe.send((e, traceback.format_exc()))

    def _send_outfiles(self, chunk_index: int, n_reads: int):
        self._write_pipe.send(chunk_index)
        self._write_pipe.send(n_reads)
        for pf in self._proxy_files:
            for chunk in pf.drain():
                self._write_pipe.send_bytes(chunk)


class OrderedChunkWriter:
    """Reorder output chunks according to input order.

    We may receive chunks of processed data from worker processes
    in any order. This class writes them to an output file in
    the correct order.
    """

    def __init__(self, outfile):
        """Initialize an OrderedChunkWriter."""
        self._chunks = dict()
        self._current_index = 0
        self._outfile = outfile

    def write(self, data: bytes, index: int):
        """Write a chunk of data to the output handler.

        :param data: The data to write.
        :param index: An integer indicating the order of the data.
        """
        self._chunks[index] = data
        while self._current_index in self._chunks:
            self._outfile.write(self._chunks[self._current_index])
            del self._chunks[self._current_index]
            self._current_index += 1

    def wrote_everything(self):
        """Return True if all chunks have been written."""
        return not self._chunks


class PipelineRunner(ABC):
    """A read processing pipeline."""

    @abstractmethod
    def run(
        self, pipeline: Pipeline, progress: Progress, outfiles: OutputFiles
    ) -> Statistics:
        """Run a pipeline on this runner.

        :param pipeline: The pipeline to run.
        :param progress: Use an object that supports .update() and .close() such
            as DummyProgress, cutadapt.utils.Progress or a tqdm instance.
        :param outfiles: The output files.
        """

    @abstractmethod
    def close(self):
        """Close all open file-descriptors."""
        pass

    @abstractmethod
    def input_file_format(self) -> FileFormat:
        """Return the input file format of the input file(s)."""
        pass

    def __enter__(self):
        """Initialize the runner."""
        return self

    def __exit__(self, *args):
        """Close the runner."""
        self.close()


class WorkerException(Exception):
    """An exception that occurred in a worker process.

    Embeds the original exception and the traceback string.
    """

    def __init__(self, wrapped_exception, tb_str):
        """Initialize a WorkerException."""
        self.e = wrapped_exception
        self.tb_str = tb_str

    def __str__(self):
        """Return a string representation of the exception."""
        return f"{self.e}\nwith traceback:\n{self.tb_str}"


StatisticsClass = typing.TypeVar("StatisticsClass", bound=Statistics)


class ParallelPipelineRunner(PipelineRunner, typing.Generic[StatisticsClass]):
    """Run a Pipeline in parallel.

    - At construction, a reader process is spawned.
    - When run() is called, as many worker processes as requested are spawned.
    - In the main process, results are written to the output files in the correct
      order, and statistics are aggregated.

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
        buffer_size: Optional[int] = None,
        statistics_class: StatisticsClass = Statistics,
    ):
        """Initialize a ParallelPipelineRunner.

        :param inpaths: The input files.
        :param n_workers: The number of worker processes to use.
        :param buffer_size: The size of the buffer used for reading the input files.
        :param statistics_class: The class to use for collecting statistics.
        """
        self._n_workers = n_workers
        self._need_work_queue: multiprocessing.Queue = mpctx.Queue()
        self._buffer_size = 4 * 1024**2 if buffer_size is None else buffer_size
        self._inpaths = inpaths
        # the workers read from these connections
        connections = [mpctx.Pipe(duplex=False) for _ in range(self._n_workers)]
        self._connections, connw = zip(*connections)
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
        self._input_file_format = self._try_receive(file_format_connection_r)
        self._statistics_class = statistics_class

    def _start_workers(
        self, pipeline, proxy_files
    ) -> Tuple[List[WorkerProcess], List[Connection]]:
        workers = []
        connections = []
        for index in range(self._n_workers):
            conn_r, conn_w = mpctx.Pipe(duplex=False)
            connections.append(conn_r)
            worker = WorkerProcess(
                index,
                pipeline,
                self._inpaths,
                proxy_files,
                self._connections[index],
                conn_w,
                self._need_work_queue,
                statistics_class=self._statistics_class,
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
        return workers, connections

    def run(self, pipeline, progress, outfiles: OutputFiles) -> StatisticsClass:
        """Run the pipeline on the input files.

        :param pipeline: The pipeline to run.
        :param progress: A progress object.
        :param outfiles: The output files.
        :return: The statistics object.
        """
        workers, connections = self._start_workers(pipeline, outfiles.proxy_files())
        chunk_writers = []

        for f in outfiles.binary_files():
            chunk_writers.append(OrderedChunkWriter(f))

        stats = self._statistics_class()
        while connections:
            ready_connections: List[Any] = multiprocessing.connection.wait(connections)
            for connection in ready_connections:
                chunk_index: int = self._try_receive(connection)
                if chunk_index == -1:
                    # the worker is done
                    cur_stats = self._try_receive(connection)
                    stats += cur_stats
                    connections.remove(connection)
                    continue

                number_of_reads: int = self._try_receive(connection)
                progress.update(number_of_reads)

                for writer in chunk_writers:
                    data = connection.recv_bytes()
                    writer.write(data, chunk_index)

        for writer in chunk_writers:
            assert writer.wrote_everything()

        for w in workers:
            w.join()

        self._reader_process.join()
        progress.close()
        return stats

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
        """Close all open file-descriptors."""
        pass

    def input_file_format(self) -> FileFormat:
        """Return the input file format of the input file(s)."""
        return self._input_file_format


class SerialPipelineRunner(PipelineRunner):
    """Run a Pipeline on a single core."""

    def __init__(
        self,
        infiles: InputFiles,
        statistics_class: Type[Statistics] = Statistics,
    ):
        """Initialize a SerialPipelineRunner.

        :param infiles: The input files.
        :param statistics_class: The class to use for collecting statistics.
        """
        self._infiles = infiles
        self._input_file_format = infiles
        self._statistics_class = statistics_class

    def run(
        self, pipeline: Pipeline, progress: Progress, outfiles: OutputFiles
    ) -> Statistics:
        """Run the pipeline on the input files.

        :param pipeline: The pipeline to run.
        :param progress: A progress object.
        :param outfiles: The output files.
        :return: The statistics object.
        """
        (n, total1_bp, total2_bp) = pipeline.process_reads(
            self._infiles, progress=progress
        )
        if progress is not None:
            progress.close()

        modifiers = getattr(pipeline, "_modifiers", None)
        assert modifiers is not None
        return self._statistics_class().collect(
            n, total1_bp, total2_bp, modifiers, pipeline._steps
        )  # type: ignore[attr-defined]

    def close(self):
        """Close all open file-descriptors from the `infiles`."""
        self._infiles.close()

    def input_file_format(self) -> FileFormat:
        """Return the detected file format of the input file(s)."""
        return detect_file_format(self._infiles._files[0])


def make_runner(
    inpaths: InputPaths,
    cores: int,
    buffer_size: Optional[int] = None,
    statistics_class: Type[Statistics] = Statistics,
) -> PipelineRunner:
    """Run a pipeline.

    This uses a SerialPipelineRunner if cores is 1 and a ParallelPipelineRunner otherwise.

    :param inpaths: The input files.
    :param cores: The number of cores to run the pipeline on (this is actually the number of worker
                  processes, there will be one extra process for reading the input file(s))
    :param buffer_size: Forwarded to `ParallelPipelineRunner()`. Ignored if cores is 1.
    """
    runner: PipelineRunner
    if cores > 1:
        runner = ParallelPipelineRunner(
            inpaths,
            n_workers=cores,
            buffer_size=buffer_size,
            statistics_class=statistics_class,
        )
    else:
        runner = SerialPipelineRunner(
            inpaths.open(),
            statistics_class=statistics_class,
        )

    return runner
