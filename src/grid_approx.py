from geometry import Point, Polygon, BoundingBox


def point_in_polygon_xy(px: float, py: float, polygon: Polygon) -> int:
    inside = False
    n = len(polygon)

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]

        ax, ay = a.x, a.y
        bx, by = b.x, b.y

        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        if abs(cross) < 1e-9:
            dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
            if dot <= 1e-9:
                return -1

        if (ay > py) != (by > py):
            x_intersection = (bx - ax) * (py - ay) / (by - ay) + ax
            if px < x_intersection:
                inside = not inside

    return 1 if inside else 0


def rasterize_polygon(polygon: Polygon, num_per_string: int) -> Polygon:
    bounding_box = BoundingBox(polygon)

    min_x, max_x = bounding_box.min.x, bounding_box.max.x
    min_y, max_y = bounding_box.min.y, bounding_box.max.y

    dx = (max_x - min_x) / (num_per_string - 1)
    dy = (max_y - min_y) / (num_per_string - 1)

    result = []

    x = min_x
    for i in range(num_per_string):
        y = min_y
        for j in range(num_per_string):
            status = point_in_polygon_xy(x, y, polygon)

            if status != 0:
                result.append(Point(x, y))

            y += dy
        x += dx

    return result


if __name__ == "__main__":
    polygon = [
        Point(3, 1),
        Point(5, 1),
        Point(5, 3),
        Point(4, 2),
        Point(3, 3)
        ]
    
    rasterize_polygon = rasterize_polygon(polygon, 10)
    print(rasterize_polygon)