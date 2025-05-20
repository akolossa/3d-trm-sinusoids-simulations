import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn.decomposition import PCA

def compute_cell_diameter(cell_points):
    if cell_points.shape[0] < 2:
        return 0  # Not enough points to compute diameter
    pca = PCA(n_components=3)
    pca.fit(cell_points)
    transformed_points = pca.transform(cell_points)
    max_distance = np.max(np.linalg.norm(transformed_points - transformed_points.mean(axis=0), axis=1))
    return max_distance

def compute_sinusoid_diameter(sinusoid_points):
    if sinusoid_points.shape[0] < 2:
        return 0  # Not enough points to compute diameter
    pca = PCA(n_components=1) # PCA to find the major axis of the sinusoid
    pca.fit(sinusoid_points)
    diameter = 2 * np.max(np.abs(pca.transform(sinusoid_points)))  #The diameter is based on the length of the component along the major axis. Twice the distance along the major axis
    return diameter


# def compute_sinusoid_diameter(sinusoid_points, directions, cell_number):
#     if sinusoid_points.shape[0] < 2:
#         return 0  # Not enough points to compute diameter
#     # Project the sinusoid points onto a plane perpendicular to the direction vector
#     sinusoid_points_centered = sinusoid_points - np.mean(sinusoid_points, axis=0)
#     projection_matrix = np.eye(cell_number) - np.outer(directions, directions)
#     projected_points = sinusoid_points_centered @ projection_matrix.T

#     # Use PCA to find the major axis of the projected points
#     pca = PCA(n_components=1)
#     pca.fit(projected_points)
#     diameter = 2 * np.max(np.abs(pca.transform(projected_points)))  # Twice the distance along the major axis

#     return diameter

def compute_cell_volume(cell_points, voxel_volume):  #!!!!!!!!!!!!!ADDED
    return len(cell_points) * voxel_volume  # Compute volume as number of voxels times voxel volume

def compute_surface_area(cell_points, state):  #!!!!!!!!!!!!!ADDED
    surface_voxels = 0
    for point in cell_points:
        neighbors = [
            (point[0]+dx, point[1]+dy, point[2]+dz)
            for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        ]
        if any(state[tuple(n)] != state[tuple(point)] for n in neighbors if 0 <= n[0] < state.shape[0] and 0 <= n[1] < state.shape[1] and 0 <= n[2] < state.shape[2]):
            surface_voxels += 1
    return surface_voxels

def compute_elongation(cell_points):  #!!!!!!!!!!!!!ADDED
    if len(cell_points) < 3:
        return 0
    pca = PCA(n_components=3)
    pca.fit(cell_points)
    components = np.sqrt(pca.explained_variance_)
    return components[0] / components[1] if components[1] > 0 else 1

def compute_contact_area_with_sinusoids(cell_points, liver):  #!!!!!!!!!!!!!ADDED
    contact_voxels = sum(liver[tuple(point)] > 0 for point in cell_points)
    return contact_voxels

def track_metrics(sim, state, liver, centroids, previous_centroids, tot_displacement, n_displacement, timepoint, metrics_csv_file, voxel_volume=1):  #!!!!!!!!!!!!!ADDED voxel_volume
    """
    Track the displacement of centroids, calculate sinusoid diameters, cell diameters, and append metrics for each cell.
    """
    cellids = state % 2**24
    types = state // 2**24
    metrics = []

    # Track centroids
    new_centroids = centroids[1:]  # Remove the first cell which initializes sinusoids
    #print(new_centroids)
    new_centroids_list = new_centroids.tolist()  # Convert to list to store results
    displacements = np.sum((new_centroids - previous_centroids)**2, axis=1)
    #print('DISPLACEMENTS:',displacements)
    tot_displacement += displacements
    n_displacement += 1
    previous_centroids = new_centroids
    avg_displacement = tot_displacement / n_displacement
    avg_displacement_list = avg_displacement.tolist()  # Convert to list to store results

    speeds = np.sqrt(displacements)  # Compute speed 
    #print('SPEED:',speeds)
    directions = displacements / np.linalg.norm(displacements)  # Compute direction as unit vector
    #print('DIRECTIONS:',directions)
    displacements_list = displacements.tolist()
    speeds_list = speeds.tolist()
    directions_list = directions.tolist()
    
    for cell_id in range(1, len(centroids)):
        # Track cell diameter
        # cell_points = np.array(np.where(cellids == cell_id)).T
        # cell_diameter = compute_cell_diameter(cell_points) if cell_points.size > 0 else 0
        # print('CELL DIAMETER:',cell_diameter)
        #!!!!!!!!!!!!!ADDED: Compute additional metrics
        # cell_volume = compute_cell_volume(cell_points, voxel_volume) if cell_points.size > 0 else 0
        # cell_surface_area = compute_surface_area(cell_points, state) if cell_points.size > 0 else 0
        # cell_elongation = compute_elongation(cell_points) if cell_points.size > 0 else 0
        # contact_area_with_sinusoids = compute_contact_area_with_sinusoids(cell_points, liver) if cell_points.size > 0 else 0
        
        # Track sinusoid diameter
        # sinusoid_diameter = 0
        # if cell_points.size > 0:
        #     sinusoid_indices = liver[cell_points[:, 0], cell_points[:, 1], cell_points[:, 2]] > 0
        #     if sinusoid_indices.any():
        #         sinusoid_points = cell_points[sinusoid_indices]
        #         sinusoid_diameter = compute_sinusoid_diameter(sinusoid_points)  
        #         print('SINUSOID DIAMETER:',sinusoid_diameter)
        metrics.append({
            "timepoint": timepoint,
            "cell_id": cell_id,
            "centroids_i": new_centroids_list[cell_id - 1],
            "avg_displacement": avg_displacement_list[cell_id - 1],
            # "cell_diameter": cell_diameter,
            # "sinusoid_diameter": sinusoid_diameter,
            "displacement": displacements_list[cell_id - 1],
            "speed": speeds_list[cell_id - 1],
            "direction": directions_list[cell_id - 1],
            # "cell_volume": cell_volume,  #!!!!!!!!!!!!!ADDED
            # "cell_surface_area": cell_surface_area,  #!!!!!!!!!!!!!ADDED
            # "cell_elongation": cell_elongation,  #!!!!!!!!!!!!!ADDED
            # "contact_area_with_sinusoids": contact_area_with_sinusoids  #!!!!!!!!!!!!!ADDED
        })

    # Convert metrics to DataFrame and append to CSV file
    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_csv_file, mode='a', header=not pd.io.common.file_exists(metrics_csv_file), index=False)

    return previous_centroids, tot_displacement, n_displacement, metrics
