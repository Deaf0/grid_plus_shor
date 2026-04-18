from typing import Tuple, List
from geometry import Point, Polygon, BoundingBox
from geometry_utils import shift_polygon
from hausdorff_distance import hausdorff_with_witness


def grid_search(A: Polygon, B: Polygon, Q0: BoundingBox, steps: int) -> Tuple[Point, float, List[Tuple[float, float, float]]]:
    xmin, xmax = Q0[0], Q0[1]
    ymin, ymax = Q0[2], Q0[3] 
    
    best_x = None
    best_val = float('inf')
    
    dx = (xmax - xmin) * (1 / steps)
    dy = (ymax - ymin) * (1 / steps)
    
    results = []  
    
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = xmin + i * dx
            y = ymin + j * dy
            
            shift = Point(x, y)
            B_shifted = shift_polygon(B, shift)
            
            val, _ = hausdorff_with_witness(A, B_shifted)
            
            results.append((x, y, val))
            
            if val < best_val:
                best_val = val
                best_x = shift
    
    return best_x, best_val, results