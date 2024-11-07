# File: RRTPlanner.py
import numpy as np
from typing import List, Tuple, Optional

from RRTGraph import RRTGraph, Edge


class CircleCollisionChecker:
    """Collision checker for circular obstacles"""

    def __init__(self, centers: List[Tuple[float, float]], radii: List[float]):
        if len(centers) != len(radii):
            raise ValueError("Number of centers must match number of radii")
        self.centers = np.array(centers)
        self.radii = np.array(radii)

    def checkCollision(self, point: np.ndarray) -> bool:
        """Check if point collides with any obstacle"""
        for center, radius in zip(self.centers, self.radii):
            if self._isInsideCircle(point, center, radius):
                return True
        return False

    def _isInsideCircle(self, point: np.ndarray, center: np.ndarray, radius: float) -> bool:
        """Check if point is inside circle"""
        return np.linalg.norm(point - center) <= radius


class RRTPlanner:
    """RRT (Rapidly-exploring Random Tree) planning algorithm"""

    def __init__(self,
                 stateBounds: List[Tuple[float, float]],
                 collisionChecker: CircleCollisionChecker,
                 stepSize: float = 0.1,
                 maxIterations: int = 1000,
                 goalSampleRate: float = 0.1):
        self.stateBounds = stateBounds
        self.collisionChecker = collisionChecker
        self.stepSize = stepSize
        self.maxIterations = maxIterations
        self.goalSampleRate = goalSampleRate
        self.graph = RRTGraph()

    def plan(self, startState: np.ndarray, goalState: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """Plan a path from start to goal state"""
        startId = self.graph.addVertex(startState)

        for _ in range(self.maxIterations):
            if np.random.random() < self.goalSampleRate:
                randomState = goalState
            else:
                randomState = self._sampleRandomState()

            nearestId = self.graph.getNearestVertex(
                randomState,
                lambda s1, s2: np.linalg.norm(s1 - s2)
            )

            newState = self._extend(self.graph.vertices[nearestId], randomState)

            if newState is not None and not self.collisionChecker.checkCollision(newState):
                newId = self.graph.addVertex(newState)
                edge = Edge(self.graph.vertices[nearestId], newState)
                self.graph.addEdge(nearestId, newId, edge)

                if np.linalg.norm(newState - goalState) < self.stepSize:
                    if not self._checkPathCollision(newState, goalState):
                        goalId = self.graph.addVertex(goalState)
                        goalEdge = Edge(newState, goalState)
                        self.graph.addEdge(newId, goalId, goalEdge)
                        return self.graph.getPath(startId, goalId), True

        return [], False

    def _sampleRandomState(self) -> np.ndarray:
        """Sample random state within bounds"""
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.stateBounds
        ])

    def _extend(self, fromState: np.ndarray, toState: np.ndarray) -> Optional[np.ndarray]:
        """Extend from one state towards another"""
        direction = toState - fromState
        distance = np.linalg.norm(direction)

        if distance < self.stepSize:
            return toState

        return fromState + (direction / distance) * self.stepSize

    def _checkPathCollision(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if path between states collides with obstacles"""
        direction = state2 - state1
        distance = np.linalg.norm(direction)
        steps = max(int(distance / self.stepSize), 1)

        for i in range(steps + 1):
            t = i / steps
            state = state1 + t * direction
            if self.collisionChecker.checkCollision(state):
                return True
        return False