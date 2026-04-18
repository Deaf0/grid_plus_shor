from typing import List, Tuple
from geometry_utils import  closest_point_on_segment
from geometry import Point, Polygon


def hausdorff_with_witness(A: Polygon, B: Polygon) -> Tuple[float, List[Tuple[Point, Point]]]:
    max_dist = 0.0
    witness = None

    for a in A:
        best_dist = float("inf")
        best_point = None

        for i in range(len(B)):
            segment_start = B[i]
            segment_end = B[(i + 1)%len(B)]

            closest_point = closest_point_on_segment(a, segment_start, segment_end)
            distance = (a - closest_point).norm()

            if distance < best_dist:
                best_dist = distance
                best_point = closest_point
        
        if best_dist > max_dist:
            max_dist = best_dist
            witness = (a, closest_point)

    for b in B:
        best_dist = float('inf')
        best_point = None
        
        for i in range(len(A)):
            segment_start = A[i]
            segment_end = A[(i+1)%len(A)]
            
            closest_point = closest_point_on_segment(b, segment_start, segment_end)
            distance = (b - closest_point).norm()
            
            if distance < best_dist:
                best_dist = distance
                best_point = closest_point
        
        if best_dist > max_dist:
            max_dist = best_dist
            witness = (best_point, b)
    
    return max_dist, witness


