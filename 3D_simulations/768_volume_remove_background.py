import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA

dimension = 768

# Load the segmented volume
liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")['image']

# Perform connected component analysis
structure = np.ones((3, 3, 3), dtype=int)  # Define 3D connectivity
labeled_array, num_features = label(liver, structure)

# Find the largest connected component
component_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background
largest_component_label = np.argmax(component_sizes) + 1

# Get voxel coordinates of the largest component
z, y, x = np.where(labeled_array == largest_component_label)
points = np.vstack((x, y, z)).T  # Shape (N,3)

# Apply PCA to find the principal directions
pca = PCA(n_components=3)
pca.fit(points)
center = points.mean(axis=0)
rotation_matrix = pca.components_

# Identify the longest central axis
longest_axis = rotation_matrix[0]  # First principal component

# Find the longest perpendicular axis
perpendicular_axis = rotation_matrix[1]  # Second principal component

# Project points onto the plane defined by the longest and perpendicular axes
projected_points = points - np.outer((points - center) @ rotation_matrix[2], rotation_matrix[2])

# Compute min/max bounds in new space
min_aligned = np.round(projected_points.min(axis=0))
max_aligned = np.round(projected_points.max(axis=0))

# Compute new shape
new_shape = (max_aligned - min_aligned + 1).astype(np.int32)
new_shape = np.maximum(new_shape, 1)  # Ensure at least 1 voxel in each dimension

# Create a new, compact volume
cropped_component = np.zeros(new_shape, dtype=bool)

# Map the points into the new volume
aligned_points_shifted = np.round(projected_points - min_aligned).astype(np.int32)
aligned_points_shifted = np.clip(aligned_points_shifted, 0, new_shape - 1)

# Assign the transformed points to the cropped volume
cropped_component[
    aligned_points_shifted[:, 2],  # Z-axis
    aligned_points_shifted[:, 1],  # Y-axis
    aligned_points_shifted[:, 0]   # X-axis
] = True

# Save using the correct filename
output_file = f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted_cropped.npz"
np.savez_compressed(output_file, image=cropped_component)

print(f"Cropped largest connected component saved to {output_file}")
print(f"New shape of the bounding box: {cropped_component.shape}")