from typing import Tuple, List
from geometry import Point, Polygon, BoundingBox
from hausdorff_distance import hausdorff_with_witness


def grid_search(A: Polygon, B: Polygon, Q0: BoundingBox, steps: int) -> Tuple[Point, float, List[Tuple[float, float, float]]]:
    xmin, xmax, ymin, ymax = Q0 
    
    best_x = None
    best_val = float('inf')
    
    inv_step = 1 / steps
    dx = (xmax - xmin) * inv_step
    dy = (ymax - ymin) * inv_step
    
    results = []  
    
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = xmin + i * dx
            y = ymin + j * dy
            
            shift = Point(x, y)    
            val, _ = hausdorff_with_witness(A, B, shift)
            
            results.append((x, y, val))
            
            if val < best_val:
                best_val = val
                best_x = shift
    
    return best_x, best_val, results