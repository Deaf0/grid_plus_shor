import numpy as np
import plotly.graph_objects as go
from geometry import Point
from q0_init import initQ0
from grid_search import grid_search
from shor_optimize import shor_optimize


A = [
    Point(0, 0),
    Point(2, 0),
    Point(2, 2),
    Point(0, 2)
]
B = [
    Point(0, 0),
    Point(2, 0),
    Point(2, 1),
    Point(1, 1),
    Point(1, 2),
    Point(0, 2)
]

Q0 = initQ0(A, B)

best_x, best_val, points = grid_search(A, B, Q0, 10)

refine_x, refine_dist = shor_optimize(A, B, best_x)

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