import os
import numpy as np
import tifffile as tiff
from skimage.transform import rescale

def downsample_image(image, downscale_factor=1.442):
    """
    Downsample the 3D image to reduce its size.
    
    Parameters:
    - image: 3D numpy array representing the image.
    - downscale_factor: Factor by which to downscale the image in each dimension.
    """
    # Downsample the image using skimage's rescale function
    downsampled_image = rescale(image, scale=(1/downscale_factor, 1/downscale_factor, 1/downscale_factor), anti_aliasing=False, preserve_range=True, order=1)
    return downsampled_image.astype(np.uint8)

# Directories for input and output
input_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
output_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
os.makedirs(output_directory, exist_ok=True)

# Loop through the images and perform downsampling
for filename in os.listdir(input_directory):
    if filename.endswith('.tiff'):
        input_path = os.path.join(input_directory, filename)
        output_filename = f'downsampled_{filename}'
        output_path = os.path.join(output_directory, output_filename)
        print(f'Processing {filename}')
        try:
            # Load the image
            image = tiff.imread(input_path)
            
            # Perform downsampling
            downsampled_image = downsample_image(image, downscale_factor=1.442)
            
            # Save the downsampled image
            tiff.imwrite(output_path, downsampled_image)
            print(f'Successfully saved downsampled image as {output_filename}')
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Downsampling complete!")