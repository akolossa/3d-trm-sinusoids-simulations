import os
import numpy as np
import tifffile as tiff
from skimage import measure, morphology

def analyze_connected_components(image):
    """
    Analyze connected components in the 3D image.
    
    Parameters:
    - image: 3D numpy array representing the binary image.
    
    Returns:
    - component_sizes: List of sizes of connected components sorted from largest to smallest.
    """
    # Label connected components in 3D
    labeled_image, num_features = measure.label(image, return_num=True, connectivity=3)
    print(f'Number of connected components: {num_features}')
    
    # Calculate the size of each component
    component_sizes = np.bincount(labeled_image.ravel())
    
    # Ignore the background component (label 0)
    component_sizes = component_sizes[1:]
    
    # Sort component sizes from largest to smallest
    component_sizes_sorted = np.sort(component_sizes)[::-1]
    
    return component_sizes_sorted

# Input path
input_path = '/home/arawa/Segmentation_shabaz_FV/final_image/cca51062_cca4000_extruded_skeleton_scaled_downsampled_inference_FI.tiff'
# Load the 3D TIFF image
image = tiff.imread(input_path)

# Convert to binary image if necessary
binary_image = image > 0

# Analyze connected components
component_sizes_sorted = analyze_connected_components(binary_image)

# Display the sizes of connected components
print("Connected components (sorted from largest to smallest):")
for i, size in enumerate(component_sizes_sorted):
    print(f"Component {i + 1}: {size} voxels")