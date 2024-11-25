import random
from Graph import GraphCC, EdgeStraight3D
import numpy as np

class PRM3D:
    def __init__(self, workspace_bounds, drone_radius, num_samples=1000, k_nearest=10):
        """
        Initialize 3D PRM for drone navigation

        Args:
            workspace_bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            drone_radius: Radius of the drone's bounding sphere
            num_samples: Number of random samples to generate
            k_nearest: Number of nearest neighbors to connect
        """
        self.bounds = workspace_bounds
        self.drone_radius = drone_radius
        self.num_samples = num_samples
        self.k_nearest = k_nearest
        self.graph = GraphCC()

    def build(self, collision_checker):
        """Build the roadmap"""
        # Generate random samples
        for _ in range(self.num_samples):
            config = self._sample_random_config()
            if not self._is_in_collision(config, collision_checker):
                self.graph.add_vertex(config)

        # Connect vertices
        vertices = self.graph.get_vertices()
        for v1 in vertices:
            nearest = self._get_k_nearest(v1, vertices)
            for v2 in nearest:
                if not self.graph.is_same_component(v1, v2):
                    if self._check_edge(v1, v2, collision_checker):
                        edge = EdgeStraight3D(
                            self.graph.get_vertex_state(v1),
                            self.graph.get_vertex_state(v2)
                        )
                        self.graph.add_edge(v1, v2, edge)

    def query(self, start, goal, collision_checker):
        """Find path from start to goal configuration"""
        # Add start and goal to graph
        start_id = self.graph.add_vertex(start)
        goal_id = self.graph.add_vertex(goal)

        # Connect to nearest neighbors
        vertices = self.graph.get_vertices()
        for v in [start_id, goal_id]:
            nearest = self._get_k_nearest(v, vertices)
            for n in nearest:
                if self._check_edge(v, n, collision_checker):
                    edge = EdgeStraight3D(
                        self.graph.get_vertex_state(v),
                        self.graph.get_vertex_state(n)
                    )
                    self.graph.add_edge(v, n, edge)

        # Find path
        path = self.graph.get_path(start_id, goal_id)
        return path

    def _sample_random_config(self):
        """Generate random 3D configuration"""
        return np.array([
            random.uniform(self.bounds[0][0], self.bounds[0][1]),
            random.uniform(self.bounds[1][0], self.bounds[1][1]),
            random.uniform(self.bounds[2][0], self.bounds[2][1])
        ])