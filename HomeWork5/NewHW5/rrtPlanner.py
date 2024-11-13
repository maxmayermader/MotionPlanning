# File: RRTPlanner.py
import math

import numpy as np
from typing import List, Tuple, Optional
from RRTGraph import RRTGraph, Edge, DubinsEdge


class CircleCollisionChecker:
    """Collision checker for two half-circle obstacles"""

    def __init__(self, centers: List[Tuple[float, float]] = [(0, -1), (0, 1)]):
        self.centers = centers  # Centers of half circles
        self.radius = 0.8  # radius = 1 - dt, where dt = 0.2
        self.wordBounds = [[-3, 3], [-1, 1]]

    def checkWorldBounds(self, state: np.ndarray) -> bool:
        """Check if state is within world bounds"""
        x, y = state[0], state[1]
        return not (self.wordBounds[0][0] <= x <= self.wordBounds[0][1] and self.wordBounds[1][0] <= y <= self.wordBounds[1][1])


    def checkCollision(self, state: np.ndarray) -> bool:
        """Check if state collides with obstacles or is outside world bounds"""
        x, y = state[0], state[1]
        if self.checkWorldBounds(state):
            return True
        # Check half circles
        for center in self.centers:
            # dist = np.sqrt(((x - center[0]) ** 2) + ((y - center[1]) ** 2))
            dist = math.sqrt(((x - center[0]) ** 2) + ((y - center[1]) ** 2))
            if dist <= self.radius:
                return True
        return False


class RRTPlanner:
    """RRT planner for Dubins car"""

    def __init__(self,
                 stateBounds: List[Tuple[float, float]],
                 collisionChecker: CircleCollisionChecker,
                 stepSize: float = 0.1,
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

            # # Check if random state is within tolerance of existing vertices
            # for vertex_id, vertex_state in self.graph.vertices.items():
            #     if self._isWithinTolerance(randomState, vertex_state):
            #         randomState = vertex_state
            #         break


            # Find nearest vertex
            # nearestId = self.graph.getNearestVertex(
            #     randomState,
            #     lambda s1, s2: np.linalg.norm(s1[:2] - s2[:2])  # Only consider x,y for distance
            # )
            # Example usage
            nearestId, nearest_point = self.graph.getNearestVertex(
                state=randomState,
                distanceFunc=lambda a, b: np.linalg.norm(a[:2] - b[:2])
            )

            # Create Dubins path to random state
            nearestState = self.graph.vertices[nearestId]
            # edge = DubinsEdge(nearestState, randomState, self.turningRadius)

            edge = self._getNonCollidingEdge(nearestState, randomState, self.turningRadius)


            # Check path collision
            if not self._checkPathCollision(edge):
                newId = self.graph.addVertex(randomState)
                self.graph.addEdge(nearestId, newId, edge)

                if (-0.8 <= nearest_point[0] <= 0.8 and (nearest_point[1] <= -0.25 or nearest_point[1] >= 0.25)) or (
                        0.8 <= randomState[0] <= 0.8 and (randomState[1] <= -0.25 or randomState[1] >= 0.25)):
                    print("bad")
                    print(nearestState[0], nearestState[1])
                    print(randomState[0], randomState[1])
                    print(self._checkPathCollision(edge))

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
        theta = np.random.uniform(self.stateBounds[2][0], self.stateBounds[2][1])
        return np.array([x, y, theta])

    def _checkPathCollision(self, edge: DubinsEdge) -> bool:
        """Check if Dubins path collides with obstacles"""

        # Check world bounds
        configurations = edge.discretize(self.stepSize)
        for config in configurations:
            if (config[0] < self.stateBounds[0][0] or config[0] > self.stateBounds[0][1] or
                    config[1] < self.stateBounds[1][0] or config[1] > self.stateBounds[1][1]):
                return True

            if self.collisionChecker.checkCollision(config):
                return True
        return False

    def _getNonCollidingEdge(self, nearestState, randomState, turningRadius) -> Tuple[np.ndarray, np.ndarray]:
        """Find the last non-colliding configuration along the Dubins path"""
        edge = DubinsEdge(nearestState, randomState, turningRadius)
        newEdge = None
        configurations = edge.discretize(self.stepSize)
        lastCollision = None
        for i, point in enumerate(configurations):
            if not self._checkPathCollision(point):
                lastCollision = configurations[:i]
            else:
                newEdge = DubinsEdge(nearestState, lastCollision[-1], turningRadius)
                break
        return newEdge
