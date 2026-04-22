import math
from typing import List, Tuple
from geometry_utils import  closest_point_on_segment
from geometry import Point, Polygon


def hausdorff_with_witness(A: Polygon, B: Polygon, shift: Point = None) -> Tuple[float, List[Tuple[Point, Point]]]:
    max_dist_sq = 0.0
    witness = None

    for a in A:
        best_dist_sq = float("inf")
        best_point = None

        for i in range(len(B)):
            segment_start = B[i]
            segment_end = B[(i + 1) % len(B)]

            if shift is not None:
                shift_start = Point(segment_start.x + shift.x, segment_start.y + shift.y)
                shifted_end = Point(segment_end.x + shift.x, segment_end.y + shift.y)
                closest_point = closest_point_on_segment(a, shift_start, shifted_end)
            else:
                closest_point = closest_point_on_segment(a, segment_start, segment_end)

            dx = a.x - closest_point.x
            dy = a.y - closest_point.y
            distance_sq = dx * dx + dy * dy

            if distance_sq < best_dist_sq:
                best_dist_sq = distance_sq
                best_point = closest_point
        
        if best_dist_sq > max_dist_sq:
            max_dist_sq = best_dist_sq
            witness = (a, best_point)

    for b in B:
        best_dist_sq = float('inf')
        best_point = None
        
        for i in range(len(A)):
            segment_start = A[i]
            segment_end = A[(i+1)%len(A)]
            
            if shift is not None:
                bx = b.x + shift.x
                by = b.y + shift.y
                b_shifted = Point(bx, by)
                closest_point = closest_point_on_segment(b_shifted, segment_start, segment_end)
            else:
                closest_point = closest_point_on_segment(b, segment_start, segment_end)
              
            dx = b.x - closest_point.x
            dy = b.y - closest_point.y
            distance_sq = dx * dx + dy * dy
            
            if distance_sq < best_dist_sq:
                best_dist_sq = distance_sq
                best_point = closest_point
        
        if best_dist_sq > max_dist_sq:
            max_dist_sq = best_dist_sq
            witness = (best_point, b)
    
    return math.sqrt(max_dist_sq), witness


