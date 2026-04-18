from geometry import Point, Polygon


def closest_point_on_segment(point: Point, a: Point, b: Point) -> Point:
    ab = b - a
    ap = point - a
    
    ab_len_sq = ab.dot(ab)
    if ab_len_sq == 0:
        return a
    
    t = ap.dot(ab) / ab_len_sq
    t = max(0.0, min(1.0, t))
    
    proj = a + ab * t
    return proj


def shift_polygon(polygon: Polygon, x: Point) -> Polygon:
    return [p + x for p in polygon]


if __name__ == "__main__":
    polygon = [Point(0, 0), Point(1, 1)]
    x = Point(1, 1)

    polygon = shift_polygon(polygon, x)
    print(polygon)
