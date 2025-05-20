import os
import numpy as np
import tifffile as tiff

def npz_to_tiff(input_path, output_path):
    """
    Convert an NPZ file to a 3D TIFF image.
    
    Parameters:
    - input_path: Path to the input NPZ file.
    - output_path: Path to the output 3D TIFF image.
    """
    # Load the NPZ file
    data = np.load(input_path)
    image = data['image']
    
    # Save the image as a TIFF file
    tiff.imwrite(output_path, image)
    print(f'Successfully saved TIFF file as {output_path}')

# Input and output paths
input_path = '/home/arawa/Segmentation_shabaz_FV/npz_files/segmentedSinusoids.npz'  # Replace with your NPZ filename
output_path = '/home/arawa/Segmentation_shabaz_FV/final_image/your_image.tiff'  # Replace with your desired output filename

# Create the output directory if it does not exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Convert the NPZ file to a TIFF image
try:
    npz_to_tiff(input_path, output_path)
except Exception as e:
    print(f'Error while processing {input_path}: {e}')

print("Conversion to TIFF complete!")