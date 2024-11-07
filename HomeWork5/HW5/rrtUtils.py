# graphUtils.py
import math
import numpy as np


class CircularObstacle:
    """Class representing a circular obstacle"""

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def getBoundary(self):
        """Get boundary points of obstacle"""
        numPoints = 100
        theta = np.linspace(0, 2 * np.pi, numPoints)
        xCoords = self.r * np.cos(theta) + self.x
        yCoords = self.r * np.sin(theta) + self.y
        return xCoords, yCoords

    def contains(self, point):
        """Check if point is inside obstacle"""
        return (point[0] - self.x) ** 2 + (point[1] - self.y) ** 2 <= self.r ** 2


class WorldBoundary2D:
    """Class representing 2D workspace boundary"""

    def __init__(self, xlim, ylim):
        self.xMin = xlim[0]
        self.xMax = xlim[1]
        self.yMin = ylim[0]
        self.yMax = ylim[1]

    def contains(self, point):
        """Check if point is outside boundary"""
        return (point[0] < self.xMin or point[0] > self.xMax or
                point[1] < self.yMin or point[1] > self.yMax)


class Edge:
    """Base class for graph edges"""

    def __init__(self, start, end, stepSize=0.1):
        self.start = start
        self.end = end
        self.stepSize = stepSize

    def getOrigin(self):
        return self.start

    def getDestination(self):
        return self.end

    def getStepSize(self):
        return self.stepSize

    def getCost(self):
        return self.getLength()

    def getPath(self):
        return [self.start, self.end]

    def reverse(self):
        self.start, self.end = self.end, self.start


class EdgeStraight(Edge):
    """Class for straight line edges"""

    def __init__(self, start, end, stepSize=0.1):
        super().__init__(start, end, stepSize)
        self.segment = end - start
        self.length = np.linalg.norm(self.segment)
        self.tStep = min(stepSize / self.length, 1)
        self.numStates = math.ceil(self.length / stepSize) + 1

    def getDiscretizedState(self, idx):
        """Get idx-th discretized state along edge"""
        if idx == 0:
            return self.start
        if idx >= self.numStates:
            return None
        if idx == self.numStates - 1:
            return self.end
        return self.start + (idx * self.tStep) * self.segment

    def getLength(self):
        return self.length

    def split(self, t):
        """Split edge at fraction t"""
        splitPoint = self.start + t * self.segment
        return (EdgeStraight(self.start, splitPoint, self.stepSize),
                EdgeStraight(splitPoint, self.end, self.stepSize))


class Tree(Graph):
    """Tree data structure for RRT"""

    def __init__(self):
        super().__init__()  # Call parent class constructor
        self.vertices = {}  # id -> state
        self.parents = {}  # id -> parent ids
        self.edges = {}  # (id1,id2) -> (cost,edge)

    def add_vertex(self, state):
        """Add vertex with given state"""
        vid = len(self.vertices)
        self.vertices[vid] = state
        self.parents[vid] = []
        return vid

    def add_edge(self, fromId, toId, edge):
        """Add edge between vertices"""
        self.edges[(fromId, toId)] = (edge.get_cost(), edge)
        self.parents[toId].append(fromId)

    def get_vertex_state(self, vid):
        """Get state of vertex"""
        return self.vertices[vid]

    def get_path(self, startId, goalId):
        """Get path from start to goal vertex"""
        vertex_path = [goalId]
        curr = goalId
        while curr != startId:
            curr = self.parents[curr][0]
            vertex_path.insert(0, curr)
        return [self.vertices[vid] for vid in vertex_path]

    def get_nearest(self, state, distance_computator, tol):
        """Return the vertex in the swath of the graph that is closest to the given state"""
        if len(self.edges) == 0:
            return self.get_nearest_vertex(state, distance_computator)

        (nearest_edge, nearest_t) = self.get_nearest_edge(state, distance_computator)

        if nearest_t <= tol:
            return nearest_edge[0]
        if nearest_t >= 1 - tol:
            return nearest_edge[1]

        return self.split_edge(nearest_edge, nearest_t)

    def get_nearest_edge(self, state, distance_computator):
        """Return the edge that is nearest to the given state"""
        nearest_dist = float('inf')
        nearest_edge = None
        nearest_t = None

        for edge_id, (cost, edge) in self.edges.items():
            (sstar, tstar) = edge.get_nearest_point(state)
            dist = distance_computator.get_distance(sstar, state)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_edge = edge_id
                nearest_t = tstar

        return (nearest_edge, nearest_t)

    def get_nearest_vertex(self, state, distance_computator):
        """Return the id of the nearest vertex to the given state"""
        nearest_dist = float('inf')
        nearest_vertex = None

        for vertex, s in self.vertices.items():
            dist = distance_computator.get_distance(s, state)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_vertex = vertex

        return nearest_vertex

    def split_edge(self, edge_id, t):
        """Split the given edge at distance t/length"""
        edge = self.edges[edge_id][1]
        (edge1, edge2) = edge.split(t)

        self.remove_edge(edge_id)
        s = edge1.get_destination()

        v = self.add_vertex(s)
        self.add_edge(edge_id[0], v, edge1)
        self.add_edge(v, edge_id[1], edge2)

        return v

    def remove_edge(self, edge_id):
        """Remove a given edge"""
        del self.edges[edge_id]
        v1 = edge_id[0]
        v2 = edge_id[1]
        self.parents[v2].remove(v1)