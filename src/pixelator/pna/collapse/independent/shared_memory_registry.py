"""A registry for shared memory buffers and numpy arrays backed by shared memory.

SharedMemoryRegistry can register objects by name and stores all information needed
to access these objects from different processes.

Copyright © 2025 Pixelgen Technologies AB.
"""

import dataclasses
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(slots=True, frozen=True, eq=True, init=True)
class ArrayDescriptor:
    """Descriptor for a shared memory array.

    These attributes are used to recreate a numpy array from a shared memory buffer.

    Attributes:
        shape: The shape of the array.
        dtype: The data type of the array.

    """

    shape: tuple[int] | tuple[int, int]
    dtype: npt.DTypeLike


class SharedMemoryRegistry:
    """Helper class that wraps a shared memory manager but registers all allocations by name.

    The registry can be used to allocate shared memory buffers and numpy arrays backed by shared memory.
    The buffers and arrays can be retrieved by name from the registry.
    Only the shared memory buffers are stored in the registry, the numpy arrays are recreated from the
    shared memory buffers when requested.

    This makes recreating the numpy arrays from the SharedMemory objects across processes easier.
    """

    def __init__(self) -> None:
        """Initialize the SharedMemoryHelper."""
        self._manager = SharedMemoryManager()
        self._buffer_registry: dict[str, SharedMemory] = {}
        self._array_registry: dict[str, ArrayDescriptor] = {}

    def __enter__(self) -> Self:
        """Enter the context.

        The SharedMemoryManager will start a management process to handle allocation/deallocation.
        """
        self._manager = self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Self:
        """Terminate the context."""
        self._manager.__exit__(exc_type, exc_val, exc_tb)
        return self

    def read_only_view(self) -> "ReadOnlySharedMemoryRegistry":
        """Return a read-only view of the registry.

        This view can be shared across threads
        """
        return ReadOnlySharedMemoryRegistry(self)

    def allocate_buffer(self, name: str, n_bytes: int) -> SharedMemory:
        """Allocates a new SharedMemory buffer.

        Args:
            name: The name of the buffer.
            n_bytes: The number of bytes to allocate.

        Returns:
            A SharedMemory object representing the allocated buffer.

        """
        buffer = self._manager.SharedMemory(n_bytes)

        # Release the block from the registry if it already exists
        # Note that the memory will be kept alive until all other processes with
        # references to this object call .unlink() as well or the manager is shutdown.
        if name in self._buffer_registry:
            self._buffer_registry[name].unlink()

        self._buffer_registry[name] = buffer
        return buffer

    def allocate_array(
        self,
        name: str,
        shape: tuple[int] | tuple[int, int],
        dtype: npt.DTypeLike,
        zero_init: bool = True,
    ) -> npt.NDArray:
        """Allocates a new SharedMemory buffer and creates a numpy array backed by this memory.

        Args:
            name: The name to register the array under in the registry.
            shape: The shape of the array.
            dtype: The data type of the array.
            zero_init: Whether to initialize the array with zeros.

        Returns:
            A numpy array backed by shared memory.

        """
        assert 1 <= len(shape) <= 2

        dtype_obj = np.dtype(dtype)
        shm = self.allocate_buffer(name, int(np.prod(shape) * dtype_obj.itemsize))
        array: npt.NDArray = np.ndarray(shape, dtype=dtype_obj, buffer=shm.buf)

        if zero_init:
            if len(shape) == 2:
                array[:, :] = 0
            else:
                array[:] = 0

        self._array_registry[name] = ArrayDescriptor(shape, dtype)
        return array

    def get_buffer(self, name: str) -> SharedMemory | None:
        """Query the registry for a shared memory buffer by name.

        Args:
            name: The name of the buffer

        """
        return self._buffer_registry.get(name)

    def get_array(self, name: str) -> np.ndarray:
        """Query the registry for a shared memory array by name.

        The array will be recreated from the registered shared memory buffer.

        Args:
            name: The name of the array

        """
        desc = self._array_registry.get(name)
        shm = self.get_buffer(name)

        if desc is None or shm is None:
            raise KeyError(f"No array with name '{name}' found in the registry")

        count = np.prod(desc.shape)
        res = np.frombuffer(shm.buf, dtype=desc.dtype, count=count)
        res.shape = desc.shape
        return res

    def unlink_buffer(self, name) -> None:
        """Unlink a shared memory buffer by name.

        Args:
            name: The name of the buffer

        """
        buffer = self._buffer_registry.pop(name, None)
        if buffer is not None:
            buffer.unlink()


class ReadOnlySharedMemoryRegistry:
    """Read-only view of a SharedMemoryRegistry.

    This class allows to create a read-only view of a SharedMemoryRegistry that can be passed to other processes.
    The view can be used to access shared memory buffers and numpy arrays backed by shared memory.
    """

    def __init__(self, registry: SharedMemoryRegistry):
        """Initialize the read-only view.

        Args:
            registry: The SharedMemoryRegistry to create a read-only view of.

        """
        self._buffer_registry = registry._buffer_registry.copy()
        self._array_registry = registry._array_registry.copy()

    def get_buffer(self, name: str) -> SharedMemory | None:
        """Query the registry for a shared memory buffer by name.

        Args:
            name: The name of the buffer

        Returns:
            The shared memory buffer or None if no buffer with the given name is found.

        """
        return self._buffer_registry.get(name)

    def get_array(self, name: str) -> np.ndarray:
        """Query the registry for a shared memory array by name.

        The array will be recreated from the registered shared memory buffer.

        Args:
            name: The name of the array

        Returns:
            The numpy array backed by shared memory.

        """
        desc = self._array_registry.get(name)
        shm = self.get_buffer(name)

        if desc is None or shm is None:
            raise KeyError(f"No array with name '{name}' found in the registry")

        count = np.prod(desc.shape)
        res = np.frombuffer(shm.buf, dtype=desc.dtype, count=count)
        res.shape = desc.shape
        return res
