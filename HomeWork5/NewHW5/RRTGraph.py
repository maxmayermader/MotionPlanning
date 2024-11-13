# File: RRTGraph.py
import dubins
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

class DubinsEdge(Edge):
    def __init__(self, state1: np.ndarray, state2: np.ndarray, turning_radius: float = 0.5):
        super().__init__(state1, state2)
        q1 = (state1[0], state1[1], state1[2])
        q2 = (state2[0], state2[1], state2[2])
        self.path = dubins.shortest_path(q1, q2, turning_radius)
        self.length = self.path.path_length()
        self.points = None  # Store discretized points

    def discretize(self, step_size: float = 0.1):
        """Return discretized points along the Dubins path including start and end points"""
        if self.points is None:
            # Get intermediate configurations
            configurations, _ = self.path.sample_many(step_size)

            # Ensure start point is included
            if len(configurations) == 0 or not np.allclose(configurations[0], self.state1):
                configurations.insert(0, (self.state1[0], self.state1[1], self.state1[2]))

            # Ensure end point is included
            if len(configurations) == 0 or not np.allclose(configurations[-1], self.state2):
                configurations.append((self.state2[0], self.state2[1], self.state2[2]))

            self.points = np.array(configurations)
        return self.points

    def getCost(self):
        return self.length

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

    def getNearestVertex(self, state: np.ndarray, distanceFunc, step_size: float = 0.1) -> Tuple[int, np.ndarray]:
        """
        Find nearest point along any edge or vertex to the given state.
        Returns: Tuple of (nearest vertex ID, nearest point)
        """
        minDist = float('inf')
        nearestVertex = None
        nearestPoint = None

        # Check all vertices first
        for vertexId, vertexState in self.vertices.items():
            dist = distanceFunc(vertexState, state)
            if dist < minDist:
                minDist = dist
                nearestVertex = vertexId
                nearestPoint = vertexState

        # Check all edges
        for (v1_id, v2_id), (_, edge) in self.edges.items():
            if isinstance(edge, DubinsEdge):
                # Get discretized points along the Dubins path
                points = edge.discretize(step_size)
                for point in points:
                    dist = distanceFunc(point, state)
                    if dist < minDist:
                        minDist = dist
                        nearestVertex = v1_id  # Associate with start vertex of edge
                        nearestPoint = point
            else:
                # For straight-line edges, use linear interpolation
                v1_state = self.vertices[v1_id]
                v2_state = self.vertices[v2_id]

                # Project point onto line segment
                edge_vector = v2_state - v1_state
                edge_length = np.linalg.norm(edge_vector)
                if edge_length == 0:
                    continue

                edge_direction = edge_vector / edge_length
                v1_to_state = state - v1_state
                projection_length = np.dot(v1_to_state, edge_direction)

                # Clamp projection to edge segment
                projection_length = max(0, min(edge_length, projection_length))
                projected_point = v1_state + projection_length * edge_direction

                dist = distanceFunc(projected_point, state)
                if dist < minDist:
                    minDist = dist
                    nearestVertex = v1_id
                    nearestPoint = projected_point

        return nearestVertex, nearestPoint

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