import os
import numpy as np
import tifffile as tiff
from skimage import measure, morphology

def remove_small_components(image, min_size=100):
    """
    Remove small connected components from the 3D image.
    """
    # Label connected components in 3D
    labeled_image, num_features = measure.label(image, return_num=True, connectivity=3)
    print(f'Number of connected components: {num_features}')
    
    # Remove small components in 3D
    cleaned_image = morphology.remove_small_objects(labeled_image, min_size=min_size, connectivity=3)
    
    # Convert back to binary image
    cleaned_image = (cleaned_image > 0).astype(np.uint8) * 255
    return cleaned_image

# Directories for input and output
input_directory = '/home/arawa/Segmentation_shabaz_FV/FV_extruded'
output_directory = '/home/arawa/Segmentation_shabaz_FV/FV_connected_components'
os.makedirs(output_directory, exist_ok=True)

# Loop through the rescaled images
for filename in os.listdir(input_directory):
    if filename.endswith('.tiff'):
        input_path = os.path.join(input_directory, filename)
        output_filename = filename.replace('.tiff', '_cca.tiff')
        output_path = os.path.join(output_directory, output_filename)
        print(f'Processing {filename}')
        try:
            # Load the rescaled image
            rescaled_image = tiff.imread(input_path)
            
            # Perform connected component analysis and remove small components
            cleaned_image = remove_small_components(rescaled_image, min_size=100)
            
            # Save the cleaned image
            tiff.imwrite(output_path, cleaned_image)
            print(f'Successfully saved cleaned image for {filename}')
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Connected component analysis complete!")