import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import random
import dubins  # We'll use the dubins library as suggested in the homework


class Node:
    def __init__(self, x: float, y: float, theta: float):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None

"""
CSpace class representing the configuration space.
@attribute xMax: maximum x-coordinate
@attribute xMin: minimum x-coordinate
@attribute yMax: maximum y-coordinate
@attribute yMin: minimum y-coordinate
"""
class CSpace:
    def __init__(self, xMax: float, xMin: float, yMax: float, yMin: float):
        self.xMax = xMax
        self.xMin = xMin
        self.yMax = yMax
        self.yMin = yMin

class RRT:
    def __init__(self, start: tuple[float, float, float],
                 goal: tuple[float, float, float],
                 cspace: CSpace, turning_radius: float,
                 obstacles, dt: float, pGoal: float = 0.1,
                 maxIterations: int = 1000):
        self.start = start
        self.goal = goal
        self.cspace = cspace
        self.turning_radius = turning_radius  # minimum turning radius = 0.5
        self.obstacles = obstacles
        self.maxIterations = maxIterations
        self.nodes: List[Node] = []
        self.pGoal = pGoal
        self.dt = dt
        self.step_size = 0.5  # Distance for discretizing Dubins curves

    def getRandomPoint(self) -> Tuple[float, float, float]:
        x = random.uniform(self.cspace.xMin, self.cspace.xMax)
        y = random.uniform(self.cspace.yMin, self.cspace.yMax)
        theta = random.uniform(-np.pi, np.pi)
        return (x, y, theta)

    def isPathClear(self, q1: Node, q2: Node) -> bool:
        # Create Dubins path between configurations
        path = dubins.shortest_path(
            (q1.x, q1.y, q1.theta),
            (q2.x, q2.y, q2.theta),
            self.turning_radius
        )

        # Sample points along the path
        samples = path.sample_many(self.step_size)[0]

        # Check each sampled point
        for x, y, _ in samples:
            # Check world bounds
            if not (self.cspace.xMin <= x <= self.cspace.xMax and
                    self.cspace.yMin <= y <= self.cspace.yMax):
                return False

            # Check obstacles (half circles)
            if ((x - 0) ** 2 + (y - 1) ** 2 <= (1 - self.dt) ** 2 or  # Upper half-circle
                    (x - 0) ** 2 + (y + 1) ** 2 <= (1 - self.dt) ** 2):  # Lower half-circle
                return False

        return True

    def findNearest(self, point: Tuple[float, float, float]) -> Node:
        min_dist = float('inf')
        nearest_node = None

        for node in self.nodes:
            # Use SE(2) distance metric (weighted sum of position and orientation differences)
            pos_dist = np.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2)
            angle_dist = abs(node.theta - point[2]) % (2 * np.pi)
            dist = pos_dist + 0.5 * angle_dist  # Weight factor for angle difference

            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    def rrtPath(self):
        # Initialize with start configuration
        start_node = Node(self.start[0], self.start[1], self.start[2])
        self.nodes.append(start_node)

        for i in range(self.maxIterations):
            # Try to connect to goal with probability pGoal
            if random.random() < self.pGoal:
                rand_config = self.goal
            else:
                rand_config = self.getRandomPoint()

            # Find nearest node
            nearest = self.findNearest(rand_config)

            # Create Dubins path
            path = dubins.shortest_path(
                (nearest.x, nearest.y, nearest.theta),
                rand_config,
                self.turning_radius
            )

            # Get end configuration
            end_config = path.sample(path.length())[0]
            new_node = Node(end_config[0], end_config[1], end_config[2])
            new_node.parent = nearest

            # Check if path is valid
            if self.isPathClear(nearest, new_node):
                self.nodes.append(new_node)

                # Check if we can connect to goal
                if (rand_config == self.goal and
                        self.isPathClear(new_node, Node(*self.goal))):
                    goal_node = Node(*self.goal)
                    goal_node.parent = new_node
                    self.nodes.append(goal_node)
                    return self.getPath(goal_node)

        return None


def plotRRTResult(nodes: List[Node], path: List[Node], cspace: CSpace,
                  start: Tuple[float, float, float],
                  goal: Tuple[float, float, float], dt: float):
    fig, ax = plt.subplots()
    ax.set_xlim(cspace.xMin, cspace.xMax)
    ax.set_ylim(cspace.yMin, cspace.yMax)

    # Plot obstacles
    circle1 = plt.Circle((0, -1), 1 - dt, color='red', fill=False)
    circle2 = plt.Circle((0, 1), 1 - dt, color='red', fill=False)
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    # Plot tree edges using Dubins curves
    for node in nodes:
        if node.parent is not None:
            path = dubins.shortest_path(
                (node.parent.x, node.parent.y, node.parent.theta),
                (node.x, node.y, node.theta),
                0.5  # minimum turning radius
            )
            samples = path.sample_many(0.1)[0]
            xs = [x for x, y, _ in samples]
            ys = [y for x, y, _ in samples]
            plt.plot(xs, ys, 'k-', linewidth=0.5)

    # Plot path if found
    if path:
        for i in range(len(path) - 1):
            dpath = dubins.shortest_path(
                (path[i].x, path[i].y, path[i].theta),
                (path[i + 1].x, path[i + 1].y, path[i + 1].theta),
                0.5
            )
            samples = dpath.sample_many(0.1)[0]
            xs = [x for x, y, _ in samples]
            ys = [y for x, y, _ in samples]
            plt.plot(xs, ys, 'b-', linewidth=2)

    # Plot start and goal configurations with arrows to show orientation
    plt.arrow(start[0], start[1], 0.2 * np.cos(start[2]), 0.2 * np.sin(start[2]),
              head_width=0.1, color='blue')
    plt.arrow(goal[0], goal[1], 0.2 * np.cos(goal[2]), 0.2 * np.sin(goal[2]),
              head_width=0.1, color='red')

    ax.set_aspect('equal')
    plt.grid(True)
    plt.title("RRT Planning for Dubins Car")
    plt.show()


def main():
    # Problem setup from homework
    start = (-2, -0.5, 0)  # (x, y, θ)
    goal = (2, -0.5, np.pi / 2)
    dt = 0.2
    cspace = CSpace(3, -3, 1, -1)

    # Create RRT instance
    rrt = RRT(start, goal, cspace, turning_radius=0.5,
              obstacles=[(0, 1), (0, -1)], dt=dt, pGoal=0.1)

    # Find path
    path = rrt.rrtPath()

    # Plot results
    plotRRTResult(rrt.nodes, path, cspace, start, goal, dt)


if __name__ == "__main__":
    main()