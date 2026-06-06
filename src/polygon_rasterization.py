import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from geometry import Point, Polygon, BoundingBox
from polygon_hausdorff_fast import default_workers


def _points_in_polygon_batch(px: np.ndarray, py: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Векторизованный ray casting; px, py — 1D, verts — (n, 2)."""
    n = len(verts)
    inside = np.zeros(px.shape, dtype=bool)
    x = verts[:, 0]
    y = verts[:, 1]

    for i in range(n):
        j = (i + 1) % n
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        denom = yj - yi
        slope = np.zeros_like(denom, dtype=np.float64)
        np.divide(xj - xi, denom, out=slope, where=np.abs(denom) > 1e-30)
        cond = ((yi > py) != (yj > py)) & (px < slope * (py - yi) + xi)
        inside ^= cond

    return inside


def _rasterize_chunk(
    polygon: Polygon,
    verts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> List[Point]:
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    px = grid_x.ravel()
    py = grid_y.ravel()
    mask = _points_in_polygon_batch(px, py, verts)
    return [Point(float(px[k]), float(py[k])) for k in np.flatnonzero(mask)]


def rasterize_polygon(
    polygon: Polygon,
    num_per_string: int,
    *,
    workers: Optional[int] = None,
) -> Polygon:
    bounding_box = BoundingBox(polygon)
    min_x, max_x = bounding_box.min.x, bounding_box.max.x
    min_y, max_y = bounding_box.min.y, bounding_box.max.y

    xs = np.linspace(min_x, max_x, num_per_string, dtype=np.float64)
    ys = np.linspace(min_y, max_y, num_per_string, dtype=np.float64)
    verts = np.asarray([(p.x, p.y) for p in polygon], dtype=np.float64)

    n_workers = workers if workers is not None else default_workers()
    if n_workers <= 1 or num_per_string < 64:
        return _rasterize_chunk(polygon, verts, xs, ys)

    n_slices = min(n_workers, num_per_string)
    x_chunks = np.array_split(xs, n_slices)
    y_chunks = [ys] * n_slices

    with ThreadPoolExecutor(max_workers=n_slices) as pool:
        parts = pool.map(
            lambda args: _rasterize_chunk(polygon, verts, args[0], args[1]),
            zip(x_chunks, y_chunks),
        )

    result: List[Point] = []
    for part in parts:
        result.extend(part)
    return result


if __name__ == "__main__":
    polygon = [
        Point(3, 1),
        Point(5, 1),
        Point(5, 3),
        Point(4, 2),
        Point(3, 3),
    ]
    print(rasterize_polygon(polygon, 10))
