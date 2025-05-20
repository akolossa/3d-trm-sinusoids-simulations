import numpy as np
import tifffile as tiff
import os

def npz_to_tiff(input_path, output_path):
    """
    Converts a 3D image stored in an NPZ file to a TIFF file.
    
    Parameters:
    - input_path: Path to the input NPZ file.
    - output_path: Path to the output 3D TIFF image.
    """
    # Load the NPZ file
    data = np.load(input_path)
    
    # Print the keys in the NPZ file
    print(f"Keys in the NPZ file: {list(data.keys())}")
    
    # Use the correct key to access the image data
    key = 'image'  # Change this to the correct key if needed
    if key not in data:
        raise KeyError(f"'{key}' is not a file in the archive. Available keys: {list(data.keys())}")
    
    image = data[key]
    
    # Save the image as a TIFF file
    tiff.imwrite(output_path, image)
    print(f'Successfully saved TIFF file as {output_path}')

# Input and output paths
input_path = '/home/arawa/Shabaz_simulation/segmentedSinusoids_AK_FV_cropped256.npz'  
output_path = '/home/arawa/Shabaz_simulation/segmentedSinusoids_AK_FV_cropped256.tiff'  

# Create the output directory if it does not exist
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Convert the NPZ file to a TIFF image
try:
    npz_to_tiff(input_path, output_path)
except Exception as e:
    print(f"Error while processing {input_path}: {e}")

print("Conversion to TIFF complete!")