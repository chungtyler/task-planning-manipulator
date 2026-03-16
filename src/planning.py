import numpy as np
from src.geometry import *

class Obstacle:
    """
    k_r: Repulsion strength
    p_0: Minimum distance
    """
    def __init__(self, geometry, k_r, p_0):
        self.geometry = geometry
        self.k_r = k_r
        self.p_0 = p_0

    # Calculate the repulsion force between joint positions q and geometry
    def compute_F(self, q):
        F = np.zeros((q.shape[0], 3), dtype=float)
        for i, q_i in enumerate(q):
            p = max(self.geometry.get_distance(q_i), 1e-5)
            if p < self.p_0:
                grad = self.geometry.get_gradient(q_i)
                F[i, :] = self.k_r * ((1 / p) - (1 / self.p_0)) * (1 / p ** 2) * grad
        return F

class Attractor:
    """
    k_a: Attraction strength
    d_bar: Minimum distance
    """
    def __init__(self, geometry, k_a, d_bar):
        self.geometry = geometry
        self.k_a = k_a
        self.d_bar = d_bar

    # Calculate the attraction force between end effector position p and geometry
    def compute_F(self, q):
        F = np.zeros((q.shape[0], 3))
        p = q[-1]
        d = self.geometry.get_normal_d(p)
        if d <= self.d_bar:
            F[:, -1] = -self.k_a * d
        else:
            F[:, -1] = -self.k_a
        return F
    
CLASS_LOOKUP = {
    'point': Point,
    'sphere': Sphere,
    'capsule': Capsule,
    'line': Line,
    'cylinder': Cylinder,
    'plane': Plane
}

class PotentialField:
    """
    Compute the total task space (attractor) or joint space (obstacles) for a manipulator
    Setup using potential_field.yaml file to initialize parameters (attraction/repulsion strength, area of effect, geometry)
    """
    def __init__(self, objects_data):
        self.objects_data = objects_data
        self.initialize_objects()

    # Initialize obstacles from YAML file
    def initialize_obstacles(self, obstacles_data):
        obstacles = []
        if obstacles_data:
            for obstacle_data in obstacles_data:
                geometry = obstacle_data.get('geometry')
                geometry_type = geometry.get('type').lower()
                params = geometry
                geometry_object = CLASS_LOOKUP[geometry_type](params)

                k_r = obstacle_data.get('k_r')
                p_0 = obstacle_data.get('p_0')
                obstacle = Obstacle(geometry_object, k_r, p_0)
                obstacles.append(obstacle)

        return obstacles
    
    # Initialize attractors from YAML file
    def initialize_attractors(self, attractors_data):
        attractors = []
        if attractors_data:
            for attractor_data in attractors_data:
                geometry = attractor_data.get('geometry')
                geometry_type = geometry.get('type').lower()
                params = geometry
                geometry_object = CLASS_LOOKUP[geometry_type](params)

                k_a = attractor_data.get('k_a')
                d_bar = attractor_data.get('d_bar')
                attractor = Obstacle(geometry_object, k_a, d_bar)
                attractors.append(attractor)

        return attractors

    # Initialize all objects in potential field
    def initialize_objects(self):
        objects = []
        obstacles_data = self.objects_data.get('obstacles_data')
        if obstacles_data:
            obstacles = self.initialize_obstacles(obstacles_data)
            objects.extend(obstacles)

        attractors_data = self.objects_data.get('attractors_data')
        if obstacles_data:
            attractors = self.initialize_attractors(attractors_data)
            objects.extend(attractors)

        self.objects = objects

    # Compute the F vector (number of joints, [Fx, Fy, Fz]) required to be applied on the joint
    def compute_F(self, q):
        F = np.zeros((q.shape[0], 3))
        for object in self.objects:
            F += object.compute_F(q)
        return F