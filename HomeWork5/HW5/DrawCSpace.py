# File: DrawCSpace.py
import matplotlib.pyplot as plt
import numpy as np


class CSpaceDrawer:
    """
    Class for visualizing the configuration space, obstacles, and RRT paths.
    """

    def __init__(self, stateBounds: list, figsize=(10, 10)):
        """
        Initialize the visualization.

        Args:
            stateBounds: List of (min, max) bounds for each dimension
            figsize: Size of the figure (width, height)
        """
        self.stateBounds = stateBounds
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setupPlot()

    def _setupPlot(self):
        """Setup the plot with proper bounds and aspect ratio."""
        self.ax.set_xlim(self.stateBounds[0])
        self.ax.set_ylim(self.stateBounds[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

    def drawObstacles(self, centers, radii):
        """Draw circular obstacles."""
        for center, radius in zip(centers, radii):
            circle = plt.Circle(center, radius, color='red', alpha=0.3)
            self.ax.add_artist(circle)

    def drawPath(self, path, color='blue', linewidth=2):
        """Draw a path as a line."""
        if len(path) < 2:
            return
        path = np.array(path)
        self.ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth)

    def drawPoint(self, point, color='black', marker='o', size=100):
        """Draw a point."""
        self.ax.scatter(point[0], point[1], c=color, marker=marker, s=size)

    def drawGraph(self, vertices, edges):
        """Draw the RRT graph."""
        # Draw edges
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.3)

        # Draw vertices
        vertices_array = np.array(list(vertices.values()))
        self.ax.scatter(vertices_array[:, 0], vertices_array[:, 1], c='black', s=20)

    def show(self):
        """Display the plot."""
        plt.show()

    def clear(self):
        """Clear the current plot."""
        self.ax.clear()
        self._setupPlot()