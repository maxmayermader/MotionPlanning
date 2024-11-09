# File: main.py
import numpy as np
from rrtPlanner import RRTPlanner, CircleCollisionChecker
from DrawCSpace import CSpaceDrawer


def main():
    # Define the state bounds [x, y, θ]
    stateBounds = [(-3, 3), (-1, 1), (-np.pi, np.pi)]

    # Define circular obstacles - two half circles
    centers = [(0, -1), (0, 1)]  # Centers of half circles
    radii = [0.8, 0.8]  # radius = 1 - dt, where dt = 0.2

    # Create collision checker
    collisionChecker = CircleCollisionChecker(centers)

    # Create RRT planner with appropriate parameters
    planner = RRTPlanner(
        stateBounds=stateBounds,
        collisionChecker=collisionChecker,
        stepSize=0.5,
        maxIterations=2000,  # Increased iterations for better coverage
        goalSampleRate=0.1  # 10% chance to sample goal
    )

    # Define start and goal states (x, y, θ)
    startState = np.array([-2, -0.5, 0])  # Initial configuration qI
    goalState = np.array([2, -0.5, np.pi / 2])  # Goal configuration qG

    # Plan path
    path, success = planner.plan(startState, goalState)
    print(f"Success: {path}")

    # Visualize
    drawer = CSpaceDrawer(stateBounds[:2])  # Only pass x,y bounds for 2D visualization

    # Draw obstacles
    drawer.drawObstacles(centers, radii)

    # Draw start and goal configurations with orientation arrows
    drawer.drawPoint(startState, color='green', size=200)
    drawer.drawPoint(goalState, color='red', size=200)

    if success:
        # Draw the RRT tree
        drawer.drawGraph(planner.graph.vertices, planner.graph.edges.keys())

        # Draw the final path
        drawer.drawPath(path)
        print("Path found!")
        print(f"Path length: {len(path)} points")
    else:
        print("No path found!")

    # Add title and show plot
    drawer.ax.set_title("RRT Path Planning for Dubins Car", fontsize=16)
    drawer.show()


if __name__ == "__main__":
    main()