# File: DrawCSpace.py
import matplotlib.pyplot as plt
import numpy as np


class CSpaceDrawer:
    """Visualization class for configuration space"""

    def __init__(self, stateBounds: list, figsize=(10, 10)):
        self.stateBounds = stateBounds
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setupPlot()

    def _setupPlot(self):
        """Setup plot parameters"""
        self.ax.set_xlim(self.stateBounds[0])
        self.ax.set_ylim(self.stateBounds[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def drawObstacles(self, centers, radii):
        """Draw circular obstacles"""
        for center, radius in zip(centers, radii):
            circle = plt.Circle(center, radius, color='red', alpha=0.3)
            self.ax.add_artist(circle)

    def drawPath(self, path, color='blue', linewidth=2):
        """Draw solution path"""
        if len(path) < 2:
            return
        path = np.array(path)
        self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)

    def drawPoint(self, point, color='black', marker='o', size=100):
        """Draw a point"""
        self.ax.scatter(point[0], point[1], c=color, marker=marker, s=size)

    def drawGraph(self, vertices, edges):
        """Draw RRT graph"""
        # Draw edges
        for (v1, v2) in edges:
            start = vertices[v1]
            end = vertices[v2]
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.3)

        # Draw vertices
        vertices_array = np.array(list(vertices.values()))
        self.ax.scatter(vertices_array[:, 0], vertices_array[:, 1], c='black', s=20)

    def show(self):
        """Display the plot"""
        plt.show()

    def clear(self):
        """Clear current plot"""
        self.ax.clear()
        self._setupPlot()