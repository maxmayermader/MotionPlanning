# File: RRTGraph.py
import numpy as np
import math
from typing import List, Tuple, Optional

# File: RRTPlanner.py (Updated)
import numpy as np
from typing import Tuple, List, Optional


class CircleCollisionChecker:
    """
    A collision checker for circular obstacles in 2D space.
    Handles collision detection between points/paths and circular obstacles.
    """

    def __init__(self, centers: List[Tuple[float, float]], radii: List[float]):
        """
        Initialize the collision checker with circular obstacles.

        Args:
            centers: List of (x, y) coordinates for circle centers
            radii: List of radii corresponding to each circle
        """
        if len(centers) != len(radii):
            raise ValueError("Number of centers must match number of radii")

        self.centers = np.array(centers)
        self.radii = np.array(radii)

    def checkCollision(self, point: np.ndarray) -> bool:
        """
        Check if a point collides with any circular obstacle.

        Args:
            point: numpy array [x, y] representing the point to check

        Returns:
            True if collision detected, False otherwise
        """
        for center, radius in zip(self.centers, self.radii):
            if self._isInsideCircle(point, center, radius):
                return True
        return False

    def _isInsideCircle(self, point: np.ndarray, center: np.ndarray,
                        radius: float) -> bool:
        """Check if a point is inside a circle."""
        return np.linalg.norm(point - center) <= radius


class Edge:
    """Class representing an edge between two states"""

    def __init__(self, state1: np.ndarray, state2: np.ndarray):
        self.state1 = state1
        self.state2 = state2
        self.cost = np.linalg.norm(state2 - state1)

    def getCost(self):
        """Return the cost of the edge"""
        return self.cost

class RRTGraph:
    """
    A class representing the RRT (Rapidly-exploring Random Tree) graph structure.
    Handles vertex and edge management for the RRT algorithm.
    """

    def __init__(self):
        # Dictionary storing vertex states: key = vertex ID, value = vertex state
        self.vertices = {}

        # Dictionary storing parent relationships: key = vertex ID, value = list of parent IDs
        self.parents = {}

        # Dictionary storing edges: key = (vertex1_id, vertex2_id), value = (cost, edge_data)
        self.edges = {}

    def addVertex(self, state: np.ndarray) -> int:
        """
        Adds a new vertex to the graph with given state.

        Args:
            state: The state vector of the vertex

        Returns:
            The ID of the newly added vertex
        """
        vertexId = len(self.vertices)
        self.vertices[vertexId] = state
        self.parents[vertexId] = []
        return vertexId

    def addEdge(self, startId: int, endId: int, edge) -> None:
        """
        Adds a directed edge between two vertices.

        Args:
            startId: ID of the start vertex
            endId: ID of the end vertex
            edge: Edge object containing geometry and cost information
        """
        self.edges[(startId, endId)] = (edge.getCost(), edge)
        self.parents[endId].append(startId)

    def getNearestVertex(self, state: np.ndarray, distanceFunc) -> int:
        """
        Finds the nearest vertex to a given state using provided distance function.

        Args:
            state: Target state to find nearest neighbor
            distanceFunc: Function to compute distance between states

        Returns:
            ID of the nearest vertex
        """
        minDist = float('inf')
        nearestVertex = None

        for vertexId, vertexState in self.vertices.items():
            dist = distanceFunc(vertexState, state)
            if dist < minDist:
                minDist = dist
                nearestVertex = vertexId

        return nearestVertex

    def getPath(self, startId: int, goalId: int) -> List[np.ndarray]:
        """
        Reconstructs the path from start to goal vertex.

        Args:
            startId: ID of the start vertex
            goalId: ID of the goal vertex

        Returns:
            List of states forming the path
        """
        if startId not in self.vertices or goalId not in self.vertices:
            return []

        path = []
        currentId = goalId

        while currentId != startId:
            path.append(self.vertices[currentId])
            if not self.parents[currentId]:
                return []
            currentId = self.parents[currentId][0]

        path.append(self.vertices[startId])
        return path[::-1]