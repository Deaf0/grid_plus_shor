from geometry import Point


def closest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    px, py = point.x, point.y
    ax, ay = a.x, a.y
    bx, by = b.x, b.y

    ab_x = bx - ax
    ab_y = by - ay

    ap_x = px - ax
    ap_y = py - ay
    
    ab_len_sq = ab_x * ab_x + ab_y * ab_y
   
    if ab_len_sq == 0:
        return a
    
    t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    
    closest_x = ax + ab_x * t
    closest_y = ay + ab_y * t

    return Point(closest_x, closest_y)

