"""Implementation of the Ramer–Douglas–Peucker line simplification algorithm.

Based on https://github.com/fhirschmann/rdp, see license statement below.

Copyright © 2024 Pixelgen Technologies AB.
Copyright (c) 2014 Fabian Hirschmann <fabian@hirschmann.email>

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import numpy.typing as npt


def _pldist(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> npt.NDArray[np.float64]:
    """Calculate the distance of points to a line segment.

    The line segment is defined by two points, start and end.
    The distance is calculated for each point in the points array.

    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(start == end):
        return np.linalg.norm(point - start)  # type: ignore

    # normalized tangent vector
    d = np.divide(end - start, np.linalg.norm(end - start))

    # signed parallel distance components
    s = np.dot(start - point, d)
    t = np.dot(point - end, d)

    # clamped parallel distance
    h = np.max([s, t, np.zeros_like(s)])

    # perpendicular distance component, as before
    c = np.cross(point - start, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)


def _ramer_douglas_peucker_iterative(
    M: np.ndarray, start_index: int, last_index: int, epsilon: float
) -> np.ndarray:
    # Initialize a stack to keep track of the ranges that need to be processed
    stack = []
    stack.append([start_index, last_index])

    # Create mask to mark wich points to keep or remove
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stack:
        start_index, last_index = stack.pop()

        # Skip empty ranges
        if start_index + 1 == last_index:
            continue

        start, end = M[start_index], M[last_index]

        index_range = np.arange(start_index + 1, last_index)
        index_range_mask = np.nonzero(indices[start_index + 1 : last_index])[0]
        masked_index_range = index_range[index_range_mask]

        # Find the distances of each point to the line segment start <-> end
        dists = _pldist(M[masked_index_range], start, end)

        # map local index for a subrange of points back to the global index
        local_max_index = np.argmax(dists)
        max_distance = dists[local_max_index]
        max_index = masked_index_range[local_max_index]

        # The point is further than epsilon to the line segment
        # split the range and add the new ranges to the stack
        if max_distance > epsilon:
            stack.append([start_index, max_index])
            stack.append([max_index, last_index])

        # The point is closer than epsilon to the line segment, remove it
        else:
            indices[start_index + 1 : last_index] = False

    return indices


def simplify_line_rdp(
    coordinates: np.ndarray, epsilon: float = 1e-2, return_mask=False
):
    """Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

    .. example:

    >>> simplify_line_rdp(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
    [[1, 1], [4, 4]]

    :param M: a series of points
    :type M: numpy array with shape ``(n,d)`` where ``n`` is the number of points and ``d`` their dimension
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    :param algo: either ``iter`` for an iterative algorithm or ``rec`` for a recursive algorithm
    :type algo: string
    :param return_mask: return mask instead of simplified array
    :type return_mask: bool
    """
    mask = _ramer_douglas_peucker_iterative(
        coordinates, 0, len(coordinates) - 1, epsilon
    )

    if return_mask:
        return mask

    return coordinates[mask]
