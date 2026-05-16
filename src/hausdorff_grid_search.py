from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from geometry import Point, Polygon
from polygon_hausdorff_fast import default_workers, hausdorff_with_witness, to_array


def _eval_shift(
    sx: float,
    sy: float,
    A: Polygon,
    B: Polygon,
    tree_A: cKDTree,
    tree_B: cKDTree,
    A_arr: np.ndarray,
    B_arr: np.ndarray,
) -> Tuple[float, float, float]:
    shift = Point(sx, sy)
    val, _, _ = hausdorff_with_witness(
        A, B, tree_A, tree_B, shift, A_arr=A_arr, B_arr=B_arr
    )
    return sx, sy, val


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
    xmin, xmax, ymin, ymax = Q0
    A_arr = to_array(A)
    B_arr = to_array(B)

    xs = np.linspace(xmin, xmax, steps + 1, dtype=np.float64)
    ys = np.linspace(ymin, ymax, steps + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()

    n_workers = workers if workers is not None else default_workers()

    if n_workers <= 1:
        results = [
            _eval_shift(float(sx), float(sy), A, B, tree_A, tree_B, A_arr, B_arr)
            for sx, sy in zip(flat_x, flat_y)
        ]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(
                pool.map(
                    lambda args: _eval_shift(
                        args[0], args[1], A, B, tree_A, tree_B, A_arr, B_arr
                    ),
                    zip(flat_x.tolist(), flat_y.tolist()),
                    chunksize=max(1, len(flat_x) // (n_workers * 4)),
                )
            )

    best_val = float("inf")
    best_x: Optional[Point] = None
    for sx, sy, val in results:
        if val < best_val:
            best_val = val
            best_x = Point(sx, sy)

    return best_x, best_val, results
