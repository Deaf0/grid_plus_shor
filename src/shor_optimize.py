import math
from typing import Optional, Tuple

from scipy.spatial import cKDTree

from geometry import Point, Polygon
from polygon_hausdorff_fast import hausdorff_with_witness, to_array


def shor_optimize(
    A: Polygon,
    B: Polygon,
    tree_A: cKDTree,
    tree_B: cKDTree,
    x0: Point,
    max_iter: int = 100,
    alpha0: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[Point, float]:
    A_arr = to_array(A)
    B_arr = to_array(B)

    x = x0.copy()
    best_x = x.copy()
    hausdorff_distance = float("inf")

    for i in range(1, max_iter + 1):
        distance, witness, source = hausdorff_with_witness(
            A, B, tree_A, tree_B, x, A_arr=A_arr, B_arr=B_arr
        )
        p, q = witness

        if distance < hausdorff_distance:
            hausdorff_distance = distance
            best_x = x.copy()

        dx = p.x - q.x
        dy = p.y - q.y
        norm_sq = dx * dx + dy * dy

        if norm_sq < eps * eps:
            break

        norm = math.sqrt(norm_sq)
        inv_norm = 1.0 / norm

        if source == "A_to_B":
            gx = dx * inv_norm
            gy = dy * inv_norm
        else:
            gx = -dx * inv_norm
            gy = -dy * inv_norm

        alpha = alpha0 / math.sqrt(i)
        x.x -= gx * alpha
        x.y -= gy * alpha

    return best_x, hausdorff_distance
