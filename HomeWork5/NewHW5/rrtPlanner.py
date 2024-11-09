# File: RRTPlanner.py
import numpy as np
from typing import List, Tuple, Optional
import dubins  # Import the dubins library for Dubins curve computation

from RRTGraph import RRTGraph, Edge



class DubinsEdge(Edge):
    def __init__(self, state1: np.ndarray, state2: np.ndarray, max_steering_angle: float = np.pi / 4):
        super().__init__(state1, state2)

        # Calculate turning radius based on maximum steering angle
        turning_radius = self.calculate_turning_radius(max_steering_angle)

        # Create dubins path between configurations
        q1 = (state1[0], state1[1], state1[2])
        q2 = (state2[0], state2[1], state2[2])
        self.path = dubins.shortest_path(q1, q2, turning_radius)
        self.length = self.path.path_length()
        self.points = None  # Store discretized points

    def discretize(self, step_size: float = 0.1):
        """Return discretized points along the Dubins path"""
        if self.points is None:
            configurations, _ = self.path.sample_many(step_size)
            self.points = np.array(configurations)
        return self.points

    def calculate_turning_radius(self, steering_angle: float, wheelbase_length: float = 1.0) -> float:
        """
        Calculate the turning radius for the Dubins car.

        Args:
            steering_angle (float): The steering angle φ in radians
            wheelbase_length (float): The wheelbase length L (default=1.0)

        Returns:
            float: The turning radius ρ
        """
        # Avoid division by zero
        if abs(steering_angle) < 1e-6:
            return float('inf')

        # Calculate turning radius: ρ = L/tan(φ)
        turning_radius = abs(wheelbase_length / np.tan(steering_angle))

        # Ensure minimum turning radius constraint (ρ_min = 0.5)
        return max(turning_radius, 0.5)

class CircleCollisionChecker:
    """Collision checker for two half-circle obstacles"""

    def __init__(self, centers: List[Tuple[float, float]] = [(0, -1), (0, 1)]):
        self.centers = centers  # Centers of half circles
        self.radius = 0.8  # radius = 1 - dt, where dt = 0.2
        self.world_bounds = [(-3, 3), (-1, 1)]  # World boundaries

    def checkCollision(self, state: np.ndarray) -> bool:
        """Check if state collides with obstacles or is outside world bounds"""
        x, y = state[0], state[1]

        # Check world bounds
        if (x < self.world_bounds[0][0] or x > self.world_bounds[0][1] or
                y < self.world_bounds[1][0] or y > self.world_bounds[1][1]):
            return True

        # Check half circles
        for center in self.centers:
            dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            if dist <= self.radius:
                return True
        return False


class RRTPlanner:
    """RRT planner for Dubins car"""

    def __init__(self,
                 stateBounds: List[Tuple[float, float]],
                 collisionChecker: CircleCollisionChecker,
                 stepSize: float = 0.5,
                 maxIterations: int = 1000,
                 goalSampleRate: float = 0.1,
                 turningRadius: float = 0.5):
        self.stateBounds = stateBounds
        self.collisionChecker = collisionChecker
        self.stepSize = stepSize
        self.maxIterations = maxIterations
        self.goalSampleRate = goalSampleRate
        # self.turningRadius = turningRadius
        self.graph = RRTGraph()
        self.tolerance = 1e-3

    def _isWithinTolerance(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if two states are within tolerance"""
        pos_diff = np.linalg.norm(state1[:2] - state2[:2])
        angle_diff = abs(state1[2] - state2[2]) % (2 * np.pi)
        return pos_diff < self.tolerance and angle_diff < self.tolerance

    def plan(self, startState: np.ndarray, goalState: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """Plan a path from start to goal state"""
        startId = self.graph.addVertex(startState)

        for _ in range(self.maxIterations):
            # Sample random state
            if np.random.random() < self.goalSampleRate:
                randomState = goalState
            else:
                randomState = self._sampleRandomState()

            # Check if random state is within tolerance of existing vertices
            for vertex_id, vertex_state in self.graph.vertices.items():
                if self._isWithinTolerance(randomState, vertex_state):
                    randomState = vertex_state
                    break


            # Find nearest vertex
            nearestId = self.graph.getNearestVertex(
                randomState,
                lambda s1, s2: np.linalg.norm(s1[:2] - s2[:2])  # Only consider x,y for distance
            )

            # Create Dubins path to random state
            nearestState = self.graph.vertices[nearestId]
            edge = DubinsEdge(nearestState, randomState)#, self.turningRadius)

            # Check path collision
            if not self._checkPathCollision(edge):
                newId = self.graph.addVertex(randomState)
                self.graph.addEdge(nearestId, newId, edge)

                # Try connecting to goal
                if np.linalg.norm(randomState[:2] - goalState[:2]) < self.stepSize:
                    goalEdge = DubinsEdge(randomState, goalState)#, self.turningRadius)
                    if not self._checkPathCollision(goalEdge):
                        goalId = self.graph.addVertex(goalState)
                        self.graph.addEdge(newId, goalId, goalEdge)
                        return self.graph.getPath(startId, goalId), True

        return [], False

    def _sampleRandomState(self) -> np.ndarray:
        """Sample random state (x, y, θ)"""
        x = np.random.uniform(self.stateBounds[0][0], self.stateBounds[0][1])
        y = np.random.uniform(self.stateBounds[1][0], self.stateBounds[1][1])
        theta = np.random.uniform(self.stateBounds[2][0], self.stateBounds[2][1])
        return np.array([x, y, theta])

    def _checkPathCollision(self, edge: DubinsEdge) -> bool:
        """Check if Dubins path collides with obstacles"""
        configurations = edge.discretize(self.stepSize)
        for config in configurations:
            if self.collisionChecker.checkCollision(config):
                return True
        return False
