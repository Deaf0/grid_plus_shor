from geometry import Point, Polygon, BoundingBox
from typing import List


def initQ0(A: Polygon, B: Polygon) -> List[float]:
    box_a = BoundingBox(A)
    box_b = BoundingBox(B)
    
    p1 = Point(
        box_a.min.x - box_b.max.x,
        box_a.min.y - box_b.max.y
    )

    p2 = Point(
        box_a.max.x - box_b.min.x,
        box_a.max.y - box_b.min.y
    )
    
    min_x = min(p1.x, p2.x)
    max_x = max(p1.x, p2.x)
    min_y = min(p1.y, p2.y)
    max_y = max(p1.y, p2.y)

    Q0 = [
        min_x,
        max_x,
        min_y,
        max_y
    ]

    return Q0