#adjust z dimensions and remove first 5 zs and last 5 zs
import os
import numpy as np
import torch
import torch.nn.functional as F
import tifffile as tiff

# Z-scaling factor based on physical voxel sizes
xy_voxel_size = 0.5682  # µm
z_voxel_size = 3.0  # µm
z_scaling_factor = xy_voxel_size / z_voxel_size  # Calculate the scaling factor

def rescale_z_dimension(image, z_scaling_factor):
    """
    Rescale the Z-dimension of the image to match physical voxel size.
    """
    # Remove the first 5 and the last 5 Z-slices
    image = image[5:-5, :, :]
    
    depth, height, width = image.shape
    new_depth = int(depth / z_scaling_factor)  # Calculate the new depth
    
    # Use interpolation to rescale Z-dimension
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    rescaled_image = F.interpolate(image, size=(new_depth, height, width), mode='trilinear', align_corners=False)
    return rescaled_image.squeeze().numpy().astype(np.uint8)

# Directories for input and output
input_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
output_directory = '/home/arawa/Segmentation_shabaz_FV/final_image'
os.makedirs(output_directory, exist_ok=True)

# Loop through the inferred images
for filename in os.listdir(input_directory):
    if filename.endswith('.tiff'):
        input_path = os.path.join(input_directory, filename)
        print(f'Processing {filename}')
        try:
            # Load the inferred image
            inferred_image = tiff.imread(input_path)
            
            # Perform Z-scaling
            scaled_image = rescale_z_dimension(inferred_image, z_scaling_factor)
            
            # Save the rescaled image with the prefix 'scaled_'
            output_filename = f'scaled_{filename}'
            output_path = os.path.join(output_directory, output_filename)
            tiff.imwrite(output_path, scaled_image)
            print(f'Successfully saved scaled image as {output_filename}')
        except Exception as e:
            print(f'Error while processing {filename}: {e}')

print("Z-scaling complete!")