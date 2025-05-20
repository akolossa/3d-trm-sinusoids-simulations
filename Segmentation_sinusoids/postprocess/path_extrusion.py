import os
import numpy as np
from skimage import measure
from skimage.draw import disk
import tifffile as tiff

def extract_skeleton_coordinates(skeleton_3d):
    # Extract coordinates of non-zero pixels in the skeletonized 3D image
    coordinates = np.argwhere(skeleton_3d == 1)
    print(f'Extracted {len(coordinates)} skeleton coordinates')
    return coordinates

def extrude_skeleton(skeleton_3d, radius=1):
    # Create an empty 3D image for the extruded volume
    extruded_image = np.zeros_like(skeleton_3d)
    
    # Get the coordinates of the skeleton
    skeleton_coords = extract_skeleton_coordinates(skeleton_3d)
    
    # For each skeleton coordinate, draw a disk in the surrounding area (to simulate extrusion)
    for coord in skeleton_coords:
        z, y, x = coord
        # Draw a disk around each skeleton point in 2D to simulate a tube in 3D
        rr, cc = disk((y, x), radius, shape=skeleton_3d.shape[1:])
        extruded_image[z, rr, cc] = 1
    
    # Debugging: Check if any disks were drawn
    if np.sum(extruded_image) == 0:
        print("Warning: No disks were drawn in the extruded image.")
    else:
        print(f"Disks drawn in the extruded image: {np.sum(extruded_image)} non-zero pixels.")
    
    return extruded_image

def save_extruded_image(extruded_image, output_path):
    tiff.imwrite(output_path, extruded_image.astype(np.uint8))

def perform_extrusion(input_path, output_path, radius=1):
    # Read the skeletonized image
    skeleton_3d = tiff.imread(input_path)
    
    # Perform extrusion
    extruded_image = extrude_skeleton(skeleton_3d, radius)
    
    # Save the extruded 3D image
    save_extruded_image(extruded_image, output_path)

# Directory containing the skeletonized TIFF images
skeletonized_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
extruded_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
os.makedirs(extruded_directory, exist_ok=True)

# Loop through the skeletonized images and perform path extrusion
for filename in os.listdir(skeletonized_directory):
    if filename.endswith('.tiff'):
        skeletonized_path = os.path.join(skeletonized_directory, filename)
        extruded_path = os.path.join(extruded_directory, f'extruded_{filename}')
        print(f'Extruding paths for {filename}')
        try:
            perform_extrusion(skeletonized_path, extruded_path, radius=2)
            print(f'Successfully saved extruded image for {filename}')
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Path extrusion complete!")