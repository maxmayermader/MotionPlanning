# File: main.py
import numpy as np
from rrtPlanner import RRTPlanner, CircleCollisionChecker
from DrawCSpace import CSpaceDrawer


def main():
    # Define the state bounds
    stateBounds = [(-10, 10), (-10, 10)]

    # Define circular obstacles
    centers = [(0, 0), (5, 5), (-5, -5)]
    radii = [2.0, 1.5, 1.0]

    # Create collision checker
    collisionChecker = CircleCollisionChecker(centers, radii)

    # Create RRT planner
    planner = RRTPlanner(
        stateBounds=stateBounds,
        collisionChecker=collisionChecker,
        stepSize=0.5,
        maxIterations=1000,
        goalSampleRate=0.1
    )

    # Define start and goal states
    startState = np.array([-8, -8])
    goalState = np.array([8, 8])

    # Plan path
    path, success = planner.plan(startState, goalState)

    # Visualize
    drawer = CSpaceDrawer(stateBounds)
    drawer.drawObstacles(centers, radii)
    drawer.drawPoint(startState, color='green', marker='*', size=200)
    drawer.drawPoint(goalState, color='red', marker='*', size=200)

    if success:
        drawer.drawPath(path)
        print("Path found!")
    else:
        print("No path found!")

    drawer.drawGraph(planner.graph.vertices, planner.graph.edges.keys())
    drawer.show()


if __name__ == "__main__":
    main()