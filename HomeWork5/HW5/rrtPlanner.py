# File: RRTPlanner.py
import numpy as np
from typing import List, Tuple, Optional
import dubins  # Import the dubins library for Dubins curve computation

from RRTGraph import RRTGraph, Edge


class DubinsEdge(Edge):
    """Edge class for Dubins car paths"""

    def __init__(self, state1: np.ndarray, state2: np.ndarray, turning_radius: float = 0.5):
        super().__init__(state1, state2)
        # Create dubins path between configurations
        q1 = (state1[0], state1[1], state1[2])
        q2 = (state2[0], state2[1], state2[2])
        self.path = dubins.shortest_path(q1, q2, turning_radius)
        self.length = self.path.path_length()

    def getCost(self):
        return self.length

    def discretize(self, step_size: float = 0.1):
        """Return discretized points along the Dubins path"""
        configurations, _ = self.path.sample_many(step_size)
        return configurations


class CircleCollisionChecker:
    """Collision checker for two half-circle obstacles"""

    def __init__(self):
        self.centers = [(0, -1), (0, 1)]  # Centers of half circles
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
        self.turningRadius = turningRadius
        self.graph = RRTGraph()

    def plan(self, startState: np.ndarray, goalState: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """Plan a path from start to goal state"""
        startId = self.graph.addVertex(startState)

        for _ in range(self.maxIterations):
            # Sample random state
            if np.random.random() < self.goalSampleRate:
                randomState = goalState
            else:
                randomState = self._sampleRandomState()

            # Find nearest vertex
            nearestId = self.graph.getNearestVertex(
                randomState,
                lambda s1, s2: np.linalg.norm(s1[:2] - s2[:2])  # Only consider x,y for distance
            )

            # Create Dubins path to random state
            nearestState = self.graph.vertices[nearestId]
            edge = DubinsEdge(nearestState, randomState, self.turningRadius)

            # Check path collision
            if not self._checkPathCollision(edge):
                newId = self.graph.addVertex(randomState)
                self.graph.addEdge(nearestId, newId, edge)

                # Try connecting to goal
                if np.linalg.norm(randomState[:2] - goalState[:2]) < self.stepSize:
                    goalEdge = DubinsEdge(randomState, goalState, self.turningRadius)
                    if not self._checkPathCollision(goalEdge):
                        goalId = self.graph.addVertex(goalState)
                        self.graph.addEdge(newId, goalId, goalEdge)
                        return self.graph.getPath(startId, goalId), True

        return [], False

    def _sampleRandomState(self) -> np.ndarray:
        """Sample random state (x, y, θ)"""
        x = np.random.uniform(self.stateBounds[0][0], self.stateBounds[0][1])
        y = np.random.uniform(self.stateBounds[1][0], self.stateBounds[1][1])
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])

    def _checkPathCollision(self, edge: DubinsEdge) -> bool:
        """Check if Dubins path collides with obstacles"""
        configurations = edge.discretize(self.stepSize)
        for config in configurations:
            if self.collisionChecker.checkCollision(config):
                return True
        return False