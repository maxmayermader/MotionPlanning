# File: DrawCSpace.py
import matplotlib.pyplot as plt
import numpy as np

from rrtPlanner import DubinsEdge


class CSpaceDrawer:
    def __init__(self, stateBounds: list, figsize=(10, 10)):
        self.stateBounds = stateBounds
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setupPlot()

    def _setupPlot(self):
        self.ax.set_xlim(self.stateBounds[0][0], self.stateBounds[0][1])
        self.ax.set_ylim(self.stateBounds[1][0], self.stateBounds[1][1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def drawObstacles(self, centers, radii):
        for center, radius in zip(centers, radii):
            if center[1] > 0:
                arc = plt.matplotlib.patches.Arc(
                    center, 2 * radius, 2 * radius,
                    theta1=0, theta2=180,
                    color='red', alpha=0.3)
            else:
                arc = plt.matplotlib.patches.Arc(
                    center, 2 * radius, 2 * radius,
                    theta1=180, theta2=360,
                    color='red', alpha=0.3)
            self.ax.add_patch(arc)
            circle = plt.Circle(center, radius, color='red', alpha=0.1)
            self.ax.add_artist(circle)

    def drawPoint(self, point, color='black', size=100):
        """Draw a configuration with correct orientation"""
        if len(point) >= 3:
            # Convert angle to degrees and adjust marker orientation
            angle_deg = np.degrees(point[2]) - 90  # Subtract 90 to align triangle with heading
            self.ax.plot(point[0], point[1],
                         marker=(3, 0, angle_deg),  # Triangle marker with specified rotation
                         color=color,
                         markersize=8,
                         markerfacecolor=color,
                         markeredgecolor=color)

    def drawGraph(self, vertices, edges):
        """Draw RRT graph with curved Dubins paths"""
        # Draw edges
        for (start_id, end_id), (_, edge) in edges.items():
            if isinstance(edge, DubinsEdge):
                # Get discretized points for the Dubins path
                points = edge.discretize(0.05)  # Smaller step size for smoother curves
                points = np.array(points)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 3)

                # Plot the curved path
                self.ax.plot(points[:, 0], points[:, 1],
                             'k-', alpha=0.3, linewidth=1)

                # Draw configurations at intervals
                if len(points) > 4:
                    step = len(points) // 4
                    for i in range(0, len(points), step):
                        self.drawPoint(points[i], color='gray')

        # Draw vertices with correct orientation
        for state in vertices.values():
            self.drawPoint(state, color='black')

    def drawPath(self, path, color='blue', linewidth=2):
        """Draw solution path with curved Dubins paths"""
        if len(path) < 2:
            return

        # Draw path segments
        for i in range(len(path) - 1):
            edge = DubinsEdge(path[i], path[i + 1], turning_radius=0.5)
            points = edge.discretize(0.05)  # Smaller step size for smoother curves
            points = np.array(points)
            if len(points.shape) == 1:
                points = points.reshape(-1, 3)

            # Plot the curved path
            self.ax.plot(points[:, 0], points[:, 1],
                         color=color, linewidth=linewidth)

            # Draw configurations along path
            if len(points) > 4:
                step = len(points) // 4
                for j in range(0, len(points), step):
                    self.drawPoint(points[j], color='green')

    def show(self):
        plt.show()

    def clear(self):
        self.ax.clear()
        self._setupPlot()
    def __init__(self, stateBounds: list, figsize=(10, 10)):
        self.stateBounds = stateBounds
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setupPlot()

    def _setupPlot(self):
        self.ax.set_xlim(self.stateBounds[0][0], self.stateBounds[0][1])
        self.ax.set_ylim(self.stateBounds[1][0], self.stateBounds[1][1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def drawObstacles(self, centers, radii):
        for center, radius in zip(centers, radii):
            if center[1] > 0:
                arc = plt.matplotlib.patches.Arc(
                    center, 2 * radius, 2 * radius,
                    theta1=0, theta2=180,
                    color='red', alpha=0.3)
            else:
                arc = plt.matplotlib.patches.Arc(
                    center, 2 * radius, 2 * radius,
                    theta1=180, theta2=360,
                    color='red', alpha=0.3)
            self.ax.add_patch(arc)
            circle = plt.Circle(center, radius, color='red', alpha=0.1)
            self.ax.add_artist(circle)

    def drawPoint(self, point, color='black', size=100):
        if len(point) == 3:
            self.ax.plot(point[0], point[1],
                         marker=(3, 0, np.degrees(point[2])),
                         markersize=8, color=color)
            arrow_length = 0.3

    def drawGraph(self, vertices, edges):
        """Draw RRT graph with Dubins paths and car positions"""
        # Draw edges
        for edge_id in edges:
            start = vertices[edge_id[0]]
            end = vertices[edge_id[1]]
            # Create a Dubins path for visualization
            edge = DubinsEdge(start, end, turning_radius=0.5)
            points = edge.discretize(0.1)

            if len(points) > 1:
                points = np.array(points)
                if len(points.shape) == 1:
                    points = points.reshape(-1, 3)

                # Plot the path
                self.ax.plot(points[:, 0], points[:, 1], 'k-', alpha=0.3)

                # Plot car positions at regular intervals
                step = len(points) // 2  # Use half the points
                if step > 0:  # Ensure we have enough points
                    for i in range(0, len(points), step):
                        self.drawPoint(points[i], color='gray', size=100)

        # Draw vertices
        for state in vertices.values():
            self.drawPoint(state, color='black', size=100)

    def drawPath(self, path, color='blue', linewidth=2):
        if len(path) < 2:
            return

        # Draw path segments using discretized points
        for i in range(len(path) - 1):
            edge = DubinsEdge(path[i], path[i + 1], turning_radius=0.5)
            points = edge.discretize(0.1)
            points = np.array(points)
            if len(points.shape) == 1:
                points = points.reshape(-1, 3)

            # Plot the discretized path
            self.ax.plot(points[:, 0], points[:, 1],
                         color=color, linewidth=linewidth)

            # Draw car configurations along path
            step = max(len(points) // 4, 1)  # Show fewer configurations
            for j in range(0, len(points), step):
                self.drawPoint(points[j], color='green', size=100)

    def show(self):
        plt.show()

    def clear(self):
        self.ax.clear()
        self._setupPlot()