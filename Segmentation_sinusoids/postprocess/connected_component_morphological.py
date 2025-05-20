# morphological closing - Combines dilation and erosion 
# to connect nearby points while maintaining the overall structure.
# then remove the less connected parts

import os
import numpy as np
import cupy as cp
import tifffile as tiff
from skimage import measure
from skimage.morphology import ball, closing, remove_small_objects

def close_skeleton(image, radius=6):
    """
    Perform morphological closing on the 3D image to connect nearby points.
    
    Parameters:
    - image: 3D CuPy array representing the binary image.
    - radius: Radius of the structuring element used for closing.
    """
    # Create a structuring element for 3D closing
    struct_elem = ball(radius)
    
    # Perform 3D closing to connect nearby points
    closed_image = closing(cp.asnumpy(image), struct_elem)
    
    return cp.array(closed_image)

def remove_small_components(image, min_size):
    """
    Remove small connected components from the 3D image.
    
    Parameters:
    - image: 3D CuPy array representing the binary image.
    - min_size: Minimum number of voxels that a connected component must have to be retained.
    """
    # Label connected components in 3D
    labeled_image, num_features = measure.label(cp.asnumpy(image), return_num=True, connectivity=3)
    print(f'Number of connected components: {num_features}')
    
    # Remove small components in 3D
    cleaned_image = remove_small_objects(labeled_image, min_size=min_size, connectivity=3)
    
    # Convert back to binary image
    cleaned_image = (cleaned_image > 0).astype(np.uint8) * 255
    return cp.array(cleaned_image)

# Directories for input and output
input_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
output_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
os.makedirs(output_directory, exist_ok=True)

# Loop through the extruded images and perform morphological closing and component removal
for filename in os.listdir(input_directory):
    if filename.endswith('.tiff'):
        input_path = os.path.join(input_directory, filename)
        output_filename = f'cca51062_{filename}'
        output_path = os.path.join(output_directory, output_filename)
        print(f'Processing {filename}')
        try:
            # Load the extruded image
            extruded_image = tiff.imread(input_path)
            
            # Convert the image to CuPy array
            extruded_image_gpu = cp.array(extruded_image)
            
            # Perform morphological closing
            closed_image_gpu = close_skeleton(extruded_image_gpu, radius=6)
            
            # Remove small components
            cleaned_image_gpu = remove_small_components(closed_image_gpu, min_size=51062)
            
            # Convert back to NumPy array
            cleaned_image = cleaned_image_gpu.get()
            
            # Save the cleaned image
            tiff.imwrite(output_path, cleaned_image)
            print(f'Successfully saved cleaned image for {filename}')
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Morphological closing and component removal complete!")
