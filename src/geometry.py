import numpy as np

"""
Used for computing the points on the surface for potential field computation

Parameters:
    r_0: Origin [x_0, y_0, z_0]
    n: Normal [a, b, c]
    r: radius
"""

class Point:
    def __init__(self, params):
        self.r_0 = np.asarray(params.get('r_0'), dtype=float)

    # Get the nearest point on the point surface to point p
    def get_nearest_point(self, _):
        p_near = self.r_0.copy()
        return p_near
    
    # Get the distance between nearest point on surface and point p
    def get_distance(self, p):
        return np.linalg.norm(p - self.get_nearest_point(p))
    
    # Get normalized gradient between nearest point on surface and point p
    def get_gradient(self, p):
        A = p - self.get_nearest_point(p)
        return A / max(1e-5, np.linalg.norm(A))
    
class Sphere(Point):
    def __init__(self, params):
        super().__init__(params)
        self.r = params.get('r')

    # Get the nearest point on the sphere surface to point p
    def get_nearest_point(self, p):
        A = p - self.r_0
        n = A / max(1e-5, np.linalg.norm(A))
        p_near = self.r_0 + self.r * n
        return p_near
    
class Capsule(Point):
    def __init__(self, params):
        super().__init__(params)
        self.r = params.get('r')
        self.a = params.get('a')

    # Get the nearest point on the capsule surface to point p
    def get_nearest_point(self, p):
        pass
    
class Line(Point):
    def __init__(self, params):
        super().__init__(params)
        n = np.asarray(params.get('n'), dtype=float)
        self.n = n / np.linalg.norm(n)

    # Get the nearest point on the line surface to point p
    def get_nearest_point(self, p):
        A = p - self.r_0
        p_near = self.r_0 + np.dot(A, self.n) * self.n
        return p_near
    
class Cylinder(Point):
    def __init__(self, params):
        super().__init__(params)
        n = np.asarray(params.get('n'), dtype=float)
        self.n = n / np.linalg.norm(n)
        self.r = params.get('r')

    # Get the nearest point on cylinder surface to point p
    def get_nearest_point(self, p):
        A = p - self.r_0
        p_near_line = self.r_0 + np.dot(A, self.n) * self.n

        B = p - p_near_line
        n = B / max(1e-5, np.linalg.norm(B))
        p_near = p_near_line + self.r * n
        return p_near
    
class Plane(Point):
    def __init__(self, params):
        super().__init__(params)
        n = np.asarray(params.get('n'), dtype=float)
        self.n = n / np.linalg.norm(n)

     # Get the nearest point on plane surface to point p
    def get_nearest_point(self, p):
        A = p - self.r_0
        p_near = p - np.dot(A, self.n) * self.n
        return p_near