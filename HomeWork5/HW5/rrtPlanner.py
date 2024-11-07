# File: RRTPlanner.py
import numpy as np
from typing import Tuple, List, Optional
from RRTGrapgh import CircleCollisionChecker, RRTGraph


class RRTPlanner:
    """
    Implementation of the RRT planning algorithm.
    """

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
        """Plan a path from start to goal state."""
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
                lambda s1, s2: np.linalg.norm(s1 - s2)
            )

            # Extend towards random state
            newState = self._extend(self.graph.vertices[nearestId], randomState)

            # Check if extension is valid
            if newState is not None and not self.collisionChecker.checkCollision(newState):
                newId = self.graph.addVertex(newState)
                self.graph.addEdge(
                    nearestId,
                    newId,
                    self._createEdge(self.graph.vertices[nearestId], newState)
                )

                # Check if we can connect to goal
                if np.linalg.norm(newState - goalState) < self.stepSize:
                    if not self._checkPathCollision(newState, goalState):
                        goalId = self.graph.addVertex(goalState)
                        self.graph.addEdge(
                            newId,
                            goalId,
                            self._createEdge(newState, goalState)
                        )
                        return self.graph.getPath(startId, goalId), True

        return [], False

    def _sampleRandomState(self) -> np.ndarray:
        """Sample a random state within the state bounds."""
        return np.array([
            np.random.uniform(low, high)
            for low, high in self.stateBounds
        ])

    def _extend(self, fromState: np.ndarray, toState: np.ndarray) -> Optional[np.ndarray]:
        """Extend from one state towards another by at most stepSize."""
        direction = toState - fromState
        distance = np.linalg.norm(direction)

        if distance < self.stepSize:
            return toState

        return fromState + (direction / distance) * self.stepSize

    def _checkPathCollision(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if path between two states collides with obstacles."""
        direction = state2 - state1
        distance = np.linalg.norm(direction)
        steps = max(int(distance / self.stepSize), 1)

        for i in range(steps + 1):
            t = i / steps
            state = state1 + t * direction
            if self.collisionChecker.checkCollision(state):
                return True
        return False

    def _createEdge(self, state1: np.ndarray, state2: np.ndarray) -> dict:
        """Create an edge between two states."""
        return {
            'cost': np.linalg.norm(state2 - state1),
            'state1': state1,
            'state2': state2
        }