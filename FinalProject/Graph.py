import math, copy
from heapq import heappush
import numpy as np

class Queue:
    """A base class for maintaining a queue"""

    def __init__(self):
        # The queue
        self.elements = []

        # The parent of each element that has been inserted into the queue
        self.parents = {}

    def __len__(self):
        """Return the length of the queue"""
        return len(self.elements)

    def insert(self, x, parent):
        """Insert an element into the queue."""
        raise NotImplementedError

    def pop(self):
        """Remove and return the first element in the queue"""
        return self.elements.pop(0)

    def get_visited(self):
        """Return the list of elements that have been inserted into the queue"""
        return list(self.parents.keys())

    def get_path(self, xI, xG):
        """Trace back parents to return a path from xI to xG"""
        path = [xG]
        x = xG
        while x != xI:
            x = self.parents.get(x)
            if x is None:
                return []
            path.insert(0, x)
        return path

    def _is_visited(self, x):
        """Return whether x has been visited (i.e., added to the queue at some point)"""
        return x in self.parents

    def _add_element_at(self, ind, x, parent):
        """Add x to the queue at index ind. Also, update the parent of x."""
        self.elements.insert(ind, x)
        self.parents[x] = parent


class QueueAstar(Queue):
    """The queue that implements the insert function for A*"""

    def __init__(self, cost_to_go_estimator):
        # For A*, we need to keep track of cost-to-come and estimated cost-to-go for each element
        super().__init__()
        self.cost_to_go_estimator = cost_to_go_estimator
        self.costs = {}

    def insert(self, x, parent, edge_cost=1):
        # The root has cost-to-come = 0, so its parent should have cost-to-come = -1
        parent_cost_to_come = -edge_cost
        if parent is not None:
            parent_cost_to_come = self.costs[parent][0]

        # Cost is a tuple (cost-to-come, cost-to-go)
        new_cost = (
            parent_cost_to_come + edge_cost,
            self.cost_to_go_estimator.get_lower_bound(x),
        )
        current_cost = self.costs.get(x)

        # Do nothing if the new cost is not smaller than the current cost
        if current_cost is not None and self._get_total_cost(
            current_cost
        ) <= self._get_total_cost(new_cost):
            return False

        # Resolve duplicate element by updating the cost of the element
        # To do this, we simply remove the element from the queue and re-insert it with the new cost
        if current_cost is not None:
            self.elements.remove(x)

        self._add_element_at(
            self._find_insert_index(self._get_total_cost(new_cost)), x, parent
        )
        self.costs[x] = new_cost
        return True

    def _get_total_cost(self, cost):
        """Return the sum of the element in cost"""
        return sum(cost)

    def _find_insert_index(self, cost):
        """Find the first index in the queue such that the cost of the corresponding element is greater than the given cost"""
        ind = 0
        while ind < len(self.elements):
            element_cost = self.costs[self.elements[ind]]
            if self._get_total_cost(element_cost) > cost:
                return ind
            ind += 1
        return ind

class Graph:
    """A class for maintaining a graph"""

    def __init__(self):
        # a dictionary whose key = id of the vertex and value = state of the vertex
        self.vertices = {}

        # a dictionary whose key = id of the vertex and value is the list of the ids of
        # its parents
        self.parents = {}

        # a dictionary whose key = (v1, v2) and value = (cost, edge).
        # v1 is the id of the origin vertex and v2 is the id of the destination vertex.
        # cost is the cost of the edge.
        # edge is of type Edge and stores information about the edge, e.g.,
        # the origin and destination states and the discretized points along the edge
        self.edges = {}

    def __str__(self):
        return "vertices: " + str(self.vertices) + " edges: " + str(self.edges)

    def add_vertex(self, state):
        """Add a vertex at a given state

        @return the id of the added vertex
        """
        vid = len(self.vertices)
        self.vertices[vid] = state
        self.parents[vid] = []
        return vid

    def get_vertex_state(self, vid):
        """Get the state of the vertex with id = vid"""
        return self.vertices[vid]

    def get_vertices(self):
        return list(self.vertices.keys())

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2"""
        self.edges[(vid1, vid2)] = (
            edge.get_cost(),
            edge,
        )
        self.parents[vid2].append(vid1)

    def remove_edge(self, edge_id):
        """Remove a given edge

        @type edge: a tuple (vid1, vid2) indicating the id of the origin and the destination vertices
        """
        del self.edges[edge_id]
        v1 = edge_id[0]
        v2 = edge_id[1]
        self.parents[v2].remove(v1)

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
        """Return the edge that is nearest to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.

        @return a tuple (nearest_edge, nearest_t) where
            * nearest_edge is a tuple (vid1, vid2), indicating the id of the origin and the destination vertices
            * nearest_t is a float in [0, 1], such that the nearest point along the edge to the given state is at
              distance nearest_t/length where length is the length of nearest_edge
        """
        nearest_dist = math.inf
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
        """Return the id of the nearest vertex to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.
        """
        nearest_dist = math.inf
        nearest_vertex = None
        for vertex, s in self.vertices.items():
            dist = distance_computator.get_distance(s, state)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_vertex = vertex
        return nearest_vertex

    def get_nearest_vertices(self, state, k, distance_computator):
        """Return the ids of k nearest vertices to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.
        """
        dist_vertices = []
        for vertex, s in self.vertices.items():
            dist = distance_computator.get_distance(s, state)
            heappush(dist_vertices, (dist, vertex))

        nearest_vertices = [
            dist_vertices[i][1] for i in range(min(k, len(dist_vertices)))
        ]
        return nearest_vertices

    def split_edge(self, edge_id, t):
        """Split the given edge at distance t/length where length is the length of the edge

        @return the id of the new vertex at the splitted point
        """
        edge = self.edges[edge_id][1]
        (edge1, edge2) = edge.split(t)

        self.remove_edge(edge_id)

        s = edge1.get_destination()
        # TODO: Ideally, we should check that edge1.get_destination() == edge2.get_origin()
        v = self.add_vertex(s)
        self.add_edge(edge_id[0], v, edge1)
        self.add_edge(v, edge_id[1], edge2)

        return v

    def get_vertex_path(self, root_vertex, goal_vertex):
        """Run Dijkstra's algorithm backward to compute the sequence of vertices from root_vertex to goal_vertex"""

        class ZeroCostToGoEstimator:
            """Cost to go estimator, which always returns 0."""

            def get_lower_bound(self, x):
                return 0

        Q = QueueAstar(ZeroCostToGoEstimator())
        Q.insert(goal_vertex, None, 0)
        while len(Q) > 0:
            v = Q.pop()
            if v == root_vertex:
                vertex_path = Q.get_path(goal_vertex, root_vertex)
                vertex_path.reverse()
                return vertex_path
            for u in self.parents[v]:
                edge_cost = self.edges[(u, v)][0]
                Q.insert(u, v, edge_cost)
        return []

    def get_path(self, root_vertex, goal_vertex):
        """Return a sequence of discretized states from root_vertex to goal_vertex"""
        vertex_path = self.get_vertex_path(root_vertex, goal_vertex)
        return self.get_path_from_vertex_path(vertex_path)

    def get_path_from_vertex_path(self, vertex_path):
        """Return a sequence of discretized states along the given vertex_path"""
        if len(vertex_path) == 0:
            return []

        path = []
        prev_vertex = vertex_path[0]
        for curr_ind in range(1, len(vertex_path)):
            curr_vertex = vertex_path[curr_ind]
            edge = self.edges[(prev_vertex, curr_vertex)][1]
            curr_path = edge.get_path()
            path.extend(curr_path)
            prev_vertex = curr_vertex

        return path

    def draw(self, ax):
        """Draw the graph on the axis ax"""
        for state in self.vertices.values():
            if (len(state)) == 2:
                ax.plot(state[0], state[1], "k.", linewidth=5)
            elif len(state) == 3:
                ax.plot(
                    state[0],
                    state[1],
                    marker=(3, 0, state[2] * 180 / math.pi - 90),
                    markersize=8,
                    linestyle="None",
                    markerfacecolor="black",
                    markeredgecolor="black",
                )

        for (_, edge) in self.edges.values():
            s2_ind = 1
            s1 = edge.get_discretized_state(s2_ind - 1)
            s2 = edge.get_discretized_state(s2_ind)
            while s2 is not None:
                ax.plot([s1[0], s2[0]], [s1[1], s2[1]], "k-", linewidth=1)
                s2_ind = s2_ind + 1
                s1 = s2
                s2 = edge.get_discretized_state(s2_ind)


class Tree(Graph):
    """A graph where each vertex has at most one parent"""

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2"""
        # Ensure that a vertex only has at most one parent (this is a tree).
        assert len(self.parents[vid2]) == 0
        super().add_edge(vid1, vid2, edge)

    def get_vertex_path(self, root_vertex, goal_vertex):
        """Trace back parents to return a path from root_vertex to goal_vertex"""
        vertex_path = [goal_vertex]
        v = goal_vertex
        while v != root_vertex:
            parents = self.parents[v]
            if len(parents) == 0:
                return []
            v = parents[0]
            vertex_path.insert(0, v)
        return vertex_path


class GraphCC(Graph):
    """An undirected graph that maintains connected components and incrementally updates it as an edge/vertex is added"""

    def __init__(self):
        super().__init__()
        self.components = []

    def get_component(self, v):
        """Return the index of the component of vertex v"""
        for ind, component in enumerate(self.components):
            if v in component:
                return ind
        raise ValueError

    def is_same_component(self, v1, v2):
        """Return whether vertices v1 and v2 are in the same connected component"""
        c1 = self.get_component(v1)
        c2 = self.get_component(v2)
        return c1 == c2

    def add_vertex(self, state):
        """Add a vertex at a given state and update the connected component

        @return the id of the added vertex
        """
        vid = super().add_vertex(state)
        self.components.append([vid])
        return vid

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2 and update the connected component"""
        reverse_edge = copy.deepcopy(edge)
        reverse_edge.reverse()
        super().add_edge(vid1, vid2, edge)
        super().add_edge(vid2, vid1, reverse_edge)

        c1 = self.get_component(vid1)
        c2 = self.get_component(vid2)

        if c1 == c2:
            return

        self.components[c1].extend(self.components[c2])
        del self.components[c2]

    def remove_edge(self, edge):
        """remove_edge is not implemented in this class"""
        raise NotImplementedError


class EdgeStraight3D():
    """3D straight line edge"""
    def __init__(self, s1, s2, step_size=0.1):
        """The constructor

        @type s1: a float indicating the state at the begining of the edge
        @type s2: a float indicating the state at the end of the edge
        @type step_size: a float indicating the length between consecutive states
            in the discretization
        """
        # The origin of the edge
        self.s1 = s1

        # The destination of the edge
        self.s2 = s2

        # The step size for discretizing the edge
        self.step_size = step_size

    def __str__(self):
        return "(" + str(self.s1) + "," + str(self.s2) + ")"

    def get_length(self):
        return np.linalg.norm(self.s2 - self.s1)

    def get_discretized_state(self, i):
        if i == 0:
            return self.s1
        t = i * self.step_size / self.get_length()
        if t > 1:
            return None
        return self.s1 + t * (self.s2 - self.s1)


class DroneCollisionChecker:
    """Collision checking for drone"""

    def __init__(self, obstacles, drone_radius):
        self.obstacles = obstacles
        self.drone_radius = drone_radius

    def check_collision(self, state):
        """Check if drone collides with obstacles"""
        for obstacle in self.obstacles:
            if self._check_sphere_collision(state, obstacle):
                return True
        return False

    def _check_sphere_collision(self, center, obstacle):
        """Check collision between drone's bounding sphere and obstacle"""
        # Implement specific collision checking based on obstacle type
        # (e.g., sphere-sphere, sphere-box collision)
        pass