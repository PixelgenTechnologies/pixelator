"""Copyright © 2025 Pixelgen Technologies AB."""

import pickle

import numpy as np

from pixelator.pna.collapse.independent.shared_memory_registry import (
    SharedMemoryRegistry,
)


def test_get_array():
    registry = SharedMemoryRegistry()
    with registry as r:
        arr = r.allocate_array("db", (1024,), dtype=np.uint8)
        arr[0:10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        same_arr = r.get_array("db")
        assert np.all(arr == same_arr)


def test_get_array_zero_init():
    registry = SharedMemoryRegistry()
    with registry as r:
        arr = r.allocate_array("db", (1024,), dtype=np.uint8, zero_init=True)
        assert np.all(arr == 0)


def test_shared_memory_pickler():
    registry = SharedMemoryRegistry()

    with registry as r:
        registry.allocate_array("test", (1024,), dtype=np.uint8)

        view = registry.read_only_view()
        bytes = pickle.dumps(view)
        assert len(bytes) > 0
