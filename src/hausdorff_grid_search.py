from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from geometry import Point, Polygon
from polygon_hausdorff_fast import hausdorff_distances_for_shifts, to_array


def find_optimal_translation_grid(
    A: Polygon,
    B: Polygon,
    tree_A: cKDTree,
    tree_B: cKDTree,
    Q0: List[float],
    steps: int,
    *,
    workers: Optional[int] = None,
) -> Tuple[Point, float, List[Tuple[float, float, float]]]:
    del workers  # сетка векторизована; параллелизм — на уровне кейсов (ProcessPool)

    xmin, xmax, ymin, ymax = Q0
    A_arr = to_array(A)
    B_arr = to_array(B)

    xs = np.linspace(xmin, xmax, steps + 1, dtype=np.float64)
    ys = np.linspace(ymin, ymax, steps + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()

    distances = hausdorff_distances_for_shifts(
        tree_A, tree_B, A_arr, B_arr, flat_x, flat_y
    )
    results = [
        (float(flat_x[i]), float(flat_y[i]), float(distances[i]))
        for i in range(distances.size)
    ]

    best_idx = int(np.argmin(distances))
    best_x = Point(float(flat_x[best_idx]), float(flat_y[best_idx]))
    best_val = float(distances[best_idx])

    return best_x, best_val, results
