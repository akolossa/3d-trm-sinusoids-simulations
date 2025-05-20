import os
import numpy as np
import tifffile as tiff

def tiff_to_npz(input_path, output_path):
    """
    Convert a 3D TIFF image to an NPZ file.
    
    Parameters:
    - input_path: Path to the input 3D TIFF image.
    - output_path: Path to the output NPZ file.
    """
    # Load the 3D TIFF image
    image = tiff.imread(input_path)
    # Save the image as an NPZ file
    np.savez_compressed(output_path, image=image)
    print(f'Successfully saved NPZ file as {output_path}')
# Input and output paths
input_path = '/home/arawa/Segmentation_shabaz_FV/final_image/your_image.tiff'  # Replace with your image filename
output_path = '/home/arawa/Segmentation_shabaz_FV/npz_files/segmentedSinusoids.npz'  # Replace with your desired output filename
# Create the output directory if it does not exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# Convert the TIFF image to an NPZ file
try:
    tiff_to_npz(input_path, output_path)
except Exception as e:
    print(f'Error while processing {input_path}: {e}')
print("Conversion to NPZ complete!")