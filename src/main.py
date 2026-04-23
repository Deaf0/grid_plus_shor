import numpy as np
import plotly.graph_objects as go
from scipy.spatial import KDTree
from geometry import Point
from q0_init import initQ0
from hausdorff_grid_search import find_optimal_translation_grid
from shor_optimize import shor_optimize
from polygon_rasterization import rasterize_polygon
from polygon_hausdorff_fast import to_array, hausdorff_with_witness


A = [
    Point(0, 0),
    Point(2, 0),
    Point(2, 2),
    Point(0, 2)
]
B = [
    Point(1, 0),
    Point(3, 0),
    Point(3, 2),
    Point(1, 2)
]

Q0 = initQ0(A, B)

A_rasterize = rasterize_polygon(A, 25)
B_rasterize = rasterize_polygon(B, 25)

A_arr = to_array(A)
B_arr = to_array(B)

A_tree = KDTree(A_arr)
B_tree = KDTree(B_arr)

best_x, best_val, points = find_optimal_translation_grid(A_rasterize, B_rasterize, A_tree, B_tree, Q0, 45)
refine_x, refine_dist = shor_optimize(A, B, A_tree, B_tree, best_x)

print(best_x, best_val)
print(refine_x, refine_dist)

x_raw = np.array([p[0] for p in points])
y_raw = np.array([p[1] for p in points])
z_raw = np.array([p[2] for p in points])

size = int(len(points)**0.5)
X = x_raw.reshape(size, size)
Y = y_raw.reshape(size, size)
Z = z_raw.reshape(size, size)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.show()