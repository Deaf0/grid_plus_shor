import math
from typing import Tuple
from geometry import Point, Polygon
from hausdorff_distance import hausdorff_with_witness
from geometry_utils import shift_polygon


def shor_optimize(A: Polygon, B: Polygon, x0: Point, max_iter: int = 100, alpha0: float = 1.0, eps: float = 1e-8) -> Tuple[Point, float]:
    x = x0.copy()
    best_x = x.copy()
    hausdorff_distance = float("inf")

    for i in range(1, max_iter + 1):
        B_shifted = shift_polygon(B, x)

        distance, (p, q) = hausdorff_with_witness(A, B_shifted)

        if distance < hausdorff_distance:
            hausdorff_distance = distance
            best_x = x.copy()

        direction = q - p
        norm = direction.norm()

        if norm < eps:
            break

        g = direction * (1 / norm)
        alpha = alpha0 / math.sqrt(i)
        x = x - g * alpha

    return best_x, hausdorff_distance