import os
import numpy as np
import tifffile as tiff
from skimage import morphology, measure
from skimage.segmentation import clear_border
from skimage.color import gray2rgb
from scipy.ndimage import convolve

def detect_junctions(skeleton):
    # Create a structuring element for detecting junctions
    struct_elem = np.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])
    filtered = convolve(skeleton.astype(np.uint8), struct_elem)
    junctions = (filtered >= 12).astype(np.uint8)  # Threshold to detect junctions
    return junctions

def skeletonize_image(image_path, output_path):
    # Load the segmented image
    binary_image = tiff.imread(image_path)  # Read the 3D binary image
    binary_image = binary_image > 0  # Convert to binary values
    skeleton_3d = np.zeros_like(binary_image)  # Initialize an array to store skeletonized image

    for z in range(binary_image.shape[0]):  # Loop through each slice (z-dimension)
        skeleton_3d[z] = morphology.skeletonize(binary_image[z])  # Apply skeletonization on each slice

    labeled_skeleton = measure.label(skeleton_3d)  # Label the skeleton to identify connected components
    labeled_skeleton = clear_border(labeled_skeleton)  # Optionally clean up border artifacts

    # Highlight junctions
    skeleton_rgb = gray2rgb(skeleton_3d.astype(np.uint8)) * 255  # Convert to RGB image and scale to [0, 255]
    for z in range(skeleton_3d.shape[0]):
        junctions = detect_junctions(skeleton_3d[z])
        skeleton_rgb[z][junctions > 0] = [128, 0, 128]  # Highlight junctions with purple color

    tiff.imwrite(output_path, skeleton_rgb.astype(np.uint8))  # Save the skeletonized image as TIFF (uint8 format)

# Directory containing the segmented TIFF images
segmented_directory = '/home/arawa/Segmentation_shabaz_FV/inference_3x3tiles'
skeletonized_directory = '/home/arawa/Segmentation_shabaz_FV/skeletonized_wjunctions'
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