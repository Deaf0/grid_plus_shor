import numpy as np
import plotly.graph_objects as go
from geometry import Point
from q0_init import initQ0
from grid_search import grid_search
from shor_optimize import shor_optimize
from grid_approx import rasterize_polygon
from hausdorff_distance import hausdorff_with_witness


A = [
    Point(0, 0),
    Point(4, 0),
    Point(4, 1),
    Point(1, 1),
    Point(1, 3),
    Point(4, 3),
    Point(4, 4),
    Point(0, 4)
]
B = [
    Point(5, 1),
    Point(7, 1),
    Point(7, 2),
    Point(6, 2),
    Point(6, 3),
    Point(7, 3),
    Point(7, 4),
    Point(5, 4)
]

Q0 = initQ0(A, B)
A_rasterize = rasterize_polygon(A, 15)
B_rasterize = rasterize_polygon(B, 15)
# dist, _ = hausdorff_with_witness(A_rasterize, B_rasterize)
# print(dist)



best_x, best_val, points = grid_search(A_rasterize, B_rasterize, Q0, 15)

# refine_x, refine_dist = shor_optimize(A, B, best_x)

print(best_x, best_val)
# print(refine_x, refine_dist)

x_raw = np.array([p[0] for p in points])
y_raw = np.array([p[1] for p in points])
z_raw = np.array([p[2] for p in points])

size = int(len(points)**0.5)
X = x_raw.reshape(size, size)
Y = y_raw.reshape(size, size)
Z = z_raw.reshape(size, size)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.show()