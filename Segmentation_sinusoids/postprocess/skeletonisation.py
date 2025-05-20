import os
import numpy as np
import tifffile as tiff
from skimage import morphology, measure
from skimage.segmentation import clear_border

def skeletonize_image(image_path, output_path):
    # Load the segmented image
    binary_image = tiff.imread(image_path)  # Read the 3D binary image
    binary_image = binary_image > 0  # Convert to binary values
    skeleton_3d = np.zeros_like(binary_image)  # Initialize an array to store skeletonized image

    for z in range(binary_image.shape[0]):  # Loop through each slice (z-dimension)
        skeleton_3d[z] = morphology.skeletonize(binary_image[z])  # Apply skeletonization on each slice

    labeled_skeleton = measure.label(skeleton_3d)  # Label the skeleton to identify connected components
    labeled_skeleton = clear_border(labeled_skeleton)  # Optionally clean up border artifacts

    tiff.imwrite(output_path, skeleton_3d.astype(np.uint8))  # Save the skeletonized image as TIFF (uint8 format)

# Directory containing the segmented TIFF images
segmented_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
skeletonized_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
os.makedirs(skeletonized_directory, exist_ok=True)

# Loop through the segmented images and perform skeletonization
for filename in os.listdir(segmented_directory):
    if filename.endswith('.tiff'):
        segmented_path = os.path.join(segmented_directory, filename)
        skeletonized_path = os.path.join(skeletonized_directory, f'skeleton_{filename}')
        print(f'Skeletonizing {filename}')
        try:
            skeletonize_image(segmented_path, skeletonized_path)
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Skeletonization complete!")