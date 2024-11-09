# File: DrawCSpace.py
import matplotlib.pyplot as plt
import numpy as np

from rrtPlanner import DubinsEdge


class CSpaceDrawer:
    """Visualization class for configuration space"""
    def __init__(self, stateBounds: list, figsize=(10, 10)):
        self.stateBounds = stateBounds
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setupPlot()

    def _setupPlot(self):
        """Setup plot parameters"""
        self.ax.set_xlim(self.stateBounds[0][0], self.stateBounds[0][1])
        self.ax.set_ylim(self.stateBounds[1][0], self.stateBounds[1][1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def drawObstacles(self, centers, radii):
        """Draw half-circle obstacles"""
        for center, radius in zip(centers, radii):
            # Create half circles
            if center[1] > 0:  # Upper half-circle
                arc = plt.matplotlib.patches.Arc(
                    center, 2*radius, 2*radius,
                    theta1=0, theta2=180,
                    color='red', alpha=0.3)
            else:  # Lower half-circle
                arc = plt.matplotlib.patches.Arc(
                    center, 2*radius, 2*radius,
                    theta1=180, theta2=360,
                    color='red', alpha=0.3)
            self.ax.add_patch(arc)
            # Fill the half circles
            circle = plt.Circle(center, radius, color='red', alpha=0.1)
            self.ax.add_artist(circle)

    # def drawPath(self, path, color='blue', linewidth=2):
    #     """Draw solution path"""
    #     if len(path) < 2:
    #         return
    #     path = np.array(path)
    #     self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)

    def drawPoint(self, point, color='black', size=100):
        """Draw a configuration with orientation"""
        if len(point) == 3:  # If point includes orientation
            # Draw position
            # self.ax.scatter(point[0], point[1], c=color, marker=(3, 0, point[2]), s=size)
            # Draw orientation arrow
            self.ax.plot(point[0], point[1], marker=(3, 0, np.degrees(point[2])), markersize=8, color=color)

            arrow_length = 0.3
            # dx = arrow_length * np.cos(point[2])
            # dy = arrow_length * np.sin(point[2])
            # self.ax.arrow(point[0], point[1], dx, dy,
            #              head_width=0.1, head_length=0.1,
            #              fc=color, ec=color)
        # else:
        #     self.ax.scatter(point[0], point[1], c=color, marker=marker, s=size)

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
        """Draw solution path with Dubins curves and car positions"""
        if len(path) < 2:
            return

        # Draw Dubins path between consecutive configurations
        for i in range(len(path) - 1):
            edge = DubinsEdge(path[i], path[i + 1], turning_radius=0.5)
            points = edge.discretize(0.1)

            # Convert points to numpy array and reshape if needed
            points = np.array(points)
            if len(points.shape) == 1:
                points = points.reshape(-1, 3)

            # Plot the path
            self.ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)

            # Plot car positions at regular intervals
            step = len(points) // 2  # Use half the points
            if step > 0:  # Ensure we have enough points
                for j in range(0, len(points), step):
                    self.drawPoint(points[j], color='green', size=150)

    def show(self):
        """Display the plot"""
        plt.show()

    def clear(self):
        """Clear current plot"""
        self.ax.clear()
        self._setupPlot()