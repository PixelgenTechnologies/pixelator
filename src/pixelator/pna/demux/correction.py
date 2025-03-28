"""BKTree implementation for fast sequence matching.

Copyright © 2024 Pixelgen Technologies AB.
"""

import numpy as np

from pixelator.pna.config import PNAAntibodyPanel


class BKTreeItem:
    """Small helper class to store the sequence and the marker ID in the BKTree."""

    __slots__ = ["id", "sequence"]

    def __init__(self, id: int, sequence: bytes):
        self.id = id
        self.sequence = sequence

    def __repr__(self):
        return f"BKTreeItem({self.id}, {self.sequence})"


from collections import deque
from operator import itemgetter

__all__ = ["hamming_distance", "BKTree"]

__version__ = "1.1"

_getitem0 = itemgetter(0)


def hamming_distance(x, y):
    """Calculate the hamming distance between two integral values.

    >>> [hamming_distance(x, 15) for x in [0, 8, 10, 12, 14, 15]]
    [4, 3, 2, 2, 1, 0]
    """
    return bin(x ^ y).count("1")


class BKTree:
    """BK-tree data structure.

    The BK-tree allows fast querying of matches that are
    "close" given a function to calculate a distance metric (e.g., Hamming
    distance or Levenshtein distance).

    Each node in the tree (including the root node) is a two-tuple of
    (item, children_dict), where children_dict is a dict whose keys are
    non-negative distances of the child to the current item and whose values
    are nodes.

    Adapter from: https://github.com/Jetsetter/pybktree
    License: MIT
    """

    def __init__(self, distance_func, items=[]):
        """Initialize a BKTree instance with given distance function.

        The distance function should be a callable that takes two items
        and returns a non-negative distance integer,

        :param distance_func: The distance function to use.
        :param items: An optional list of items to add on initialization.

        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree.distance_func is hamming_distance
        True
        >>> tree = BKTree(hamming_distance, [])
        >>> list(tree)
        []
        >>> tree = BKTree(hamming_distance, [0, 4, 5])
        >>> sorted(tree)
        [0, 4, 5]
        """
        self.distance_func = distance_func
        self.tree = None

        _add = self.add
        for item in items:
            _add(item)

    def add(self, item):
        """Add given item to this tree.

        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree.add(4)
        >>> sorted(tree)
        [4]
        >>> tree.add(15)
        >>> sorted(tree)
        [4, 15]
        """
        node = self.tree
        if node is None:
            self.tree = (item, {})
            return

        # Slight speed optimization -- avoid lookups inside the loop
        _distance_func = self.distance_func

        while True:
            parent, children = node
            distance = _distance_func(item, parent)
            node = children.get(distance)
            if node is None:
                children[distance] = (item, {})
                break

    def find(self, item, n):
        """Find items in this tree with a distance <= `n` from `item`.

         Return list of (distance, item) tuples ordered by distance.

        :param item: The item to find matches for.
        :param n: The maximum distance to consider a match.

        >>> tree = BKTree(hamming_distance)
        >>> tree.find(13, 1)
        []
        >>> tree.add(0)
        >>> tree.find(1, 1)
        [(1, 0)]
        >>> for item in [0, 4, 5, 14, 15]:
        ...     tree.add(item)
        >>> sorted(tree)
        [0, 0, 4, 5, 14, 15]
        >>> sorted(tree.find(13, 1))
        [(1, 5), (1, 15)]
        >>> sorted(tree.find(13, 2))
        [(1, 5), (1, 15), (2, 4), (2, 14)]
        >>> sorted(tree.find(0, 1000)) == [(hamming_distance(x, 0), x) for x in tree]
        True
        """
        if self.tree is None:
            return []

        candidates = deque([self.tree])
        found = []

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend
        _found_append = found.append
        _distance_func = self.distance_func

        while candidates:
            candidate, children = _candidates_popleft()
            distance = _distance_func(candidate, item)
            if distance <= n:
                _found_append((distance, candidate))

            if children:
                lower = distance - n
                upper = distance + n
                _candidates_extend(
                    c for d, c in children.items() if lower <= d <= upper
                )

        found.sort(key=_getitem0)
        return found

    def __iter__(self):
        """Return iterator over all items in this tree.

        Items are yielded in arbitrary order.

        >>> tree = BKTree(hamming_distance)
        >>> list(tree)
        []
        >>> tree = BKTree(hamming_distance, [1, 2, 3, 4, 5])
        >>> sorted(tree)
        [1, 2, 3, 4, 5]
        """
        if self.tree is None:
            return

        candidates = deque([self.tree])

        # Slight speed optimization -- avoid lookups inside the loop
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend

        while candidates:
            candidate, children = _candidates_popleft()
            yield candidate
            _candidates_extend(children.values())

    def __repr__(self):
        """Return a string representation of this BK-tree with a little bit of info.

        >>> BKTree(hamming_distance)
        <BKTree using hamming_distance with no top-level nodes>
        >>> BKTree(hamming_distance, [0, 4, 8, 14, 15])
        <BKTree using hamming_distance with 3 top-level nodes>
        """
        return "<{} using {} with {} top-level nodes>".format(
            self.__class__.__name__,
            self.distance_func.__name__,
            len(self.tree[1]) if self.tree is not None else "no",
        )


def hamming_distance_i8(s1: BKTreeItem, s2: BKTreeItem | bytes) -> int:
    """Calculate the byte-wise Hamming distance between two sequences."""
    b1 = np.frombuffer(s1.sequence, dtype=np.int8)
    b2 = np.frombuffer(s2.sequence if isinstance(s2, BKTreeItem) else s2, dtype=np.int8)
    return int(np.sum(b1 != b2))


def build_bktree(panel: PNAAntibodyPanel, sequence_key: str) -> BKTree:
    """Create a BKTree from the panel sequences.

    This allows us to quickly find the closest sequence to a given query sequence with up to a given distance.
    The distance function is the edit distance

    :param panel: The panel to build the tree from
    :param sequence_key: The key in the panel dataframe that contains the sequences
    :return: The BKTree
    """
    tree = BKTree(hamming_distance_i8)

    for _, row in panel.df.iterrows():
        seq = row[sequence_key].encode("ascii")
        tree.add(BKTreeItem(row["marker_id"], seq))

    return tree


def build_exact_dict_lookup(
    panel: PNAAntibodyPanel, sequence_key: str
) -> dict[bytes, str]:
    """Create a set from the panel sequences.

    This allows us to quickly find exact matches for a sequence in the panel.

    :param panel: The panel to build the lookup from
    :param sequence_key: The key in the panel dataframe that contains the sequences
    :return: The lookup table
    """
    lut = dict()

    for _, row in panel.df.iterrows():
        seq = row[sequence_key].encode("ascii")
        lut[seq] = row["marker_id"]

    return lut
