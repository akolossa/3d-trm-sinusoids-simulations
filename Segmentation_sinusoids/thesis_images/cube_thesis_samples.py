import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parameters
n = 10  # Number of cubes along each axis
cube_size = 1  # Size of each cube

# Generate the coordinates for the small cubes inside the grid (including depth)
x = np.arange(0, n * cube_size, cube_size)  # x coordinates
y = np.arange(0, n * cube_size, cube_size)  # y coordinates
z = np.array([0])  # z coordinate (1 layer)

# Create the grid of small cubes (meshgrid for 3D)
X, Y, Z = np.meshgrid(x, y, z)

# List of coordinates to color yellow
yellow_coords = [(0, 3), (1, 9), (2, 4), (3, 1), (5, 6), (8, 7), (4, 8), (6, 0), (7, 5), (9, 2)]


# Function to plot a cube
def plot_cube_black(ax, position, size, color):
    x, y, z = position
    vertices = np.array([[x, y, z],
                         [x + size[0], y, z],
                         [x + size[0], y + size[1], z],
                         [x, y + size[1], z],
                         [x, y, z + size[2]],
                         [x + size[0], y, z + size[2]],
                         [x + size[0], y + size[1], z + size[2]],
                         [x, y + size[1], z + size[2]]])
    faces = [[vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]],
             [vertices[j] for j in [3, 0, 4, 7]],
             [vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]]]
    poly3d = Poly3DCollection(faces, facecolors='b', linewidths=0.5, edgecolors='b', alpha=1)
    ax.add_collection3d(poly3d)

# Function to plot a cube
def plot_cube(ax, position, size, color='blue'):
    x, y, z = position
    vertices = np.array([[x, y, z],
                         [x + size[0], y, z],
                         [x + size[0], y + size[1], z],
                         [x, y + size[1], z],
                         [x, y, z + size[2]],
                         [x + size[0], y, z + size[2]],
                         [x + size[0], y + size[1], z + size[2]],
                         [x, y + size[1], z + size[2]]])
    faces = [[vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [2, 3, 7, 6]],
             [vertices[j] for j in [3, 0, 4, 7]],
             [vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]]]
    poly3d = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='r', alpha=.25)
    ax.add_collection3d(poly3d)

# Plot the cubes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot all cubes in the grid
for i in range(n):
    for j in range(n):
        if (i, j) in yellow_coords:
            # Plot the yellow cube at the base layer
            plot_cube(ax, (X[i, j, 0], Y[i, j, 0], Z[0, 0, 0]), (cube_size, cube_size, cube_size), color='yellow')
            # Insert a 1x1x0.01 layer in the middle of the Z stack for the yellow solid
            mid_z = 0.5 * cube_size - 0.005  # Middle of the Z stack
            plot_cube_black(ax, (X[i, j, 0], Y[i, j, 0], 0.4), (cube_size, cube_size, 0.01), color='black')
            plot_cube_black(ax, (X[i, j, 0], Y[i, j, 0], 0.1), (cube_size, cube_size, 0.01), color='black')
            plot_cube_black(ax, (X[i, j, 0], Y[i, j, 0], 0.8), (cube_size, cube_size, 0.01), color='black')

        else:
            plot_cube(ax, (X[i, j, 0], Y[i, j, 0], Z[0, 0, 0]), (cube_size, cube_size, cube_size), color='blue')



# ax.set_xlim([0, n * cube_size])
# ax.set_ylim([0, n * cube_size])
# ax.set_zlim([0, cube_size])

ax.set_axis_off()

# Adjust the view angle
ax.view_init(elev=60, azim=120)  # Adjust the elevation and azimuthal angle

# Save the plot
plt.savefig('/home/arawa/Segmentation_shabaz_FV/imagea.png')
plt.show()