import math
from typing import Tuple
from geometry import Point, Polygon
from hausdorff_distance import hausdorff_with_witness


def shor_optimize(
    A: Polygon, 
    B: Polygon, 
    x0: Point,  
    max_iter: int = 100, 
    alpha0: float = 1.0, 
    eps: float = 1e-8
) -> Tuple[Point, float]:

    x = x0.copy()
    best_x = x.copy()
    hausdorff_distance = float("inf")

    for i in range(1, max_iter + 1):
        distance, witness = hausdorff_with_witness(A, B, x)
        p, q = witness

        if distance < hausdorff_distance:
            hausdorff_distance = distance
            best_x = x.copy()

        dx = q.x - p.x
        dy = q.y - p.y

        norm_sq = dx * dx + dy * dy

        if norm_sq < eps * eps:
            break
        
        norm = math.sqrt(norm_sq)

        gx = dx / norm
        gy = dy / norm

        alpha = alpha0 / math.sqrt(i)
        
        x.x -= gx * alpha  
        x.y -= gy * alpha

    return best_x, hausdorff_distance