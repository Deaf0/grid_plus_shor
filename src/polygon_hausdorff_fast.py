import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from geometry import Point, Polygon


def to_array(polygon: Polygon) -> np.ndarray:
    return np.asarray([(point.x, point.y) for point in polygon], dtype=np.float64)


def hausdorff_with_witness(
    A: Polygon,
    B: Polygon,
    tree_A: cKDTree,
    tree_B: cKDTree,
    shift: Optional[Point] = None,
    *,
    A_arr: Optional[np.ndarray] = None,
    B_arr: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[Point, Point], str]:
    if A_arr is None:
        A_arr = to_array(A)
    if B_arr is None:
        B_arr = to_array(B)

    if shift is not None:
        sx, sy = shift.x, shift.y
    else:
        sx, sy = 0.0, 0.0

    offset = np.array([sx, sy], dtype=np.float64)

    dists_b, idxs_b = tree_B.query(A_arr - offset)
    i_ab = int(np.argmax(dists_b))
    max_dist_ab = float(dists_b[i_ab])

    b_shifted = B_arr[idxs_b[i_ab]] + offset
    dists_a, idxs_a = tree_A.query(B_arr + offset)
    i_ba = int(np.argmax(dists_a))
    max_dist_ba = float(dists_a[i_ba])

    if max_dist_ab >= max_dist_ba:
        a_pt = A_arr[i_ab]
        return (
            max_dist_ab,
            (Point(float(a_pt[0]), float(a_pt[1])), Point(float(b_shifted[0]), float(b_shifted[1]))),
            "A_to_B",
        )

    a_pt = A_arr[idxs_a[i_ba]]
    b_pt = B_arr[i_ba] + offset
    return (
        max_dist_ba,
        (Point(float(a_pt[0]), float(a_pt[1])), Point(float(b_pt[0]), float(b_pt[1]))),
        "B_to_A",
    )


def default_workers() -> int:
    count = os.cpu_count() or 1
    return max(1, min(count, 32))
