"""Utility functions for working with iterables and collections.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import itertools
import typing
from typing import Any, Generator, Iterable, List, Set, Union

T = typing.TypeVar("T")


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter.

    Taken from python itertools recipes.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def single_value(xs: Union[List[T], Set[T]]) -> T:
    """Extract the first value in a List or Set if the collection has a single value.

    Args:
        xs: a collection of values

    Returns:
        the first value in the collection (T)

    Raises:
        AssertionError: if the collection is empty or has more than one value
    """
    if len(xs) == 0:
        raise AssertionError("Empty collection")
    if len(xs) > 1:
        raise AssertionError("More than one element in collection")
    return list(xs)[0]


def flatten(iterable: Iterable[Iterable[Any] | Any]) -> Generator[Any, None, None]:
    """Flatten an Iterable containing items or collection of items.

    Note: only list, set, tuple are flattened, strings and bytes are yielded as is

    Args:
        iterable: list of lists or list of sets

    Returns:
        A generator yielding the flattened items (Generator[Any, None, None])

    Yields:
        the flattened items (Any)
    """
    for item in iterable:
        if isinstance(item, (str, bytes)):
            yield item
        elif isinstance(item, (list, set, tuple)):
            yield from item
        else:
            yield item
