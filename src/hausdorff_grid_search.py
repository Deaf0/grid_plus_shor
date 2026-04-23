from scipy.spatial import KDTree
from typing import Tuple, List
from geometry import Point, Polygon
from polygon_hausdorff_fast import hausdorff_with_witness


def find_optimal_translation_grid(
    A: Polygon, 
    B: Polygon, 
    tree_A: KDTree, 
    tree_B: KDTree,  
    Q0: List[float], 
    steps: int
) -> Tuple[Point, float, List[Tuple[float, float, float]]]:
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
            val, _ = hausdorff_with_witness(A, B, tree_A, tree_B, shift)
            
            results.append((x, y, val))
            
            if val < best_val:
                best_val = val
                best_x = shift
    
    return best_x, best_val, results