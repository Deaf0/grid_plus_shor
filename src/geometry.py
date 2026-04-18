import math
from typing import List


class Point:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        
        eps = 1e-10
        return (abs(self.x - other.x) < eps and
                abs(self.y - other.y) < eps)

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        return Point(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Point':
        return self.__mul__(scalar)
    
    def dot(self, other: 'Point') -> float:
        return self.x * other.x + self.y * other.y
    
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


Polygon = List[Point]


class BoundingBox:
    min: Point
    max: Point
    
    def __init__(self, poly: Polygon):
        self.min = Point(float('inf'), float('inf'))
        self.max = Point(-float('inf'), -float('inf'))
        
        for p in poly:
            self.min.x = min(self.min.x, p.x)
            self.min.y = min(self.min.y, p.y)
            self.max.x = max(self.max.x, p.x)
            self.max.y = max(self.max.y, p.y)
