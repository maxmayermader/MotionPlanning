# File: RRTGraph.py
import numpy as np
from typing import List, Tuple, Optional


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
    """A class representing the RRT graph structure"""

    def __init__(self):
        self.vertices = {}  # vertex ID -> state
        self.parents = {}  # vertex ID -> list of parent IDs
        self.edges = {}  # (vertex1_id, vertex2_id) -> (cost, edge)

    def addVertex(self, state: np.ndarray) -> int:
        """Add a vertex at a given state"""
        vertexId = len(self.vertices)
        self.vertices[vertexId] = state
        self.parents[vertexId] = []
        return vertexId

    def addEdge(self, startId: int, endId: int, edge: Edge) -> None:
        """Add an edge from vertex with id startId to vertex with id endId"""
        self.edges[(startId, endId)] = (edge.getCost(), edge)
        self.parents[endId].append(startId)

    def getNearestVertex(self, state: np.ndarray, distanceFunc) -> int:
        """Find nearest vertex to given state using provided distance function"""
        minDist = float('inf')
        nearestVertex = None

        for vertexId, vertexState in self.vertices.items():
            dist = distanceFunc(vertexState, state)
            if dist < minDist:
                minDist = dist
                nearestVertex = vertexId

        return nearestVertex

    def getPath(self, startId: int, goalId: int) -> List[np.ndarray]:
        """Get path from start to goal vertex"""
        if startId not in self.vertices or goalId not in self.vertices:
            return []

        path = [self.vertices[goalId]]
        currentId = goalId

        while currentId != startId:
            if not self.parents[currentId]:
                return []
            currentId = self.parents[currentId][0]
            path.insert(0, self.vertices[currentId])

        return path