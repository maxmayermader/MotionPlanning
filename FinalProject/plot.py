from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import matplotlib.pyplot as plt

def plot(xlim, ylim, zlim, obstacles):
    ax = plt.axes(projection='3d')
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_zlim(zlim)

    for obs in obstacles:
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)

        r = 1 * obs[3]
        x = obs[0] + r * np.sin(phi) * np.cos(theta)
        y = obs[1] + r * np.sin(phi) * np.sin(theta)
        z = obs[2] + r * np.cos(phi)
        ax.plot_surface(x,y,z, cmap='magma', alpha=0.8)

    plt.show()

# def createObs(x0,y0,z0,r0):
#     xlim = np.array((-3,3))
#     ylim = np.array((-3,3))
#     zlim = np.array((-3,3))

#     theta = np.linspace(0, 2 * np.pi, 100)
#     phi = np.linspace(0, np.pi, 50)
#     theta, phi = np.meshgrid(theta, phi)
#     r = 1 * 0.7
#     x = r * np.sin(phi) * np.cos(theta) 
#     y = r * np.sin(phi) * np.sin(theta) 
#     z = r * np.cos(phi)
#     print(f"Shape of x, y, z: {x.shape}")
#     print(f"Min and max values - x: ({x.min():.2f}, {x.max():.2f}), y: ({y.min():.2f}, {y.max():.2f}), z: ({z.min():.2f}, {z.max():.2f})")

#     ax = plt.axes(projection='3d')
#     plt.xlim(xlim)
#     plt.ylim(ylim)
#     ax.set_zlim(zlim)
    
#     # fig = plt.figure(figsize=(10, 10))
#     # ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z, cmap='magma', alpha=0.8)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Sphere')
#     ax.set_box_aspect((1, 1, 1))
#     plt.show()



obstacles = []
obs1 = (2.7,2.5,2,0.5)
obs2 = (-2.7,-1.5,-2,1)
obstacles.append(obs1)
obstacles.append(obs2)
# createObs(-1,2,-.5,.05)


xlim = np.array((-3,3))
ylim = np.array((-3,3))
zlim = np.array((-3,3))
plot(xlim, ylim, zlim, obstacles)


