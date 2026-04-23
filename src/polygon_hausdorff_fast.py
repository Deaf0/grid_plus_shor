from scipy.spatial import KDTree
from typing import List, Tuple
from geometry import Point, Polygon


def to_array(polygon: Polygon) -> List:
    return [(point.x, point.y) for point in polygon]


def hausdorff_with_witness(
    A: Polygon, 
    B: Polygon, 
    tree_A: KDTree, 
    tree_B: KDTree, 
    shift: Point = None
) -> Tuple[float, Tuple[Point, Point]]:
    max_dist = 0.0
    witness = None

    if shift is not None:
        sx, sy = shift.x, shift.y
    else:
        sx, sy = 0, 0

    for a in A:
        query_point = (a.x - sx, a.y - sy)
        dist, idx = tree_B.query(query_point)
        b = B[idx]

        b_shifted = Point(b.x + sx, b.y + sy)

        if dist > max_dist:
            max_dist = dist
            witness = (a, b_shifted)

    for b in B:
        b_shifted = Point(b.x + sx, b.y + sy)
        query_point = (b_shifted.x, b_shifted.y)

        dist, idx = tree_A.query(query_point)
        a = A[idx]

        if dist > max_dist:
            max_dist = dist
            witness = (a, b_shifted)
    
    return max_dist, witness


