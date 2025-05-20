import numpy as np
from PIL import Image
import tifffile as tiff
import os

# Load the TIFF file
tiff_file = '/home/arawa/Segmentation_shabaz_FV/raw_fiji_images.tiff/C1-finalimage.tif'  # Update with your file path
z_stack = tiff.imread(tiff_file)  # Read the Z-stack (269 images)
num_slices, height, width = z_stack.shape  # shape should be (269, 1857, 1857) # Check dimensions
grid_size = 6
tile_height = height // grid_size
tile_width = width // grid_size

# Create directory to save images if it doesn't exist
output_dir = '/home/arawa/Segmentation_shabaz_FV/3x3_tiles'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the grid
for i in range(grid_size):
    for j in range(grid_size):
        # Calculate coordinates for the tiles
        start_x = j * tile_width
        end_x = start_x + tile_width
        start_y = i * tile_height
        end_y = start_y + tile_height

        # Slice the Z-stack for the current tile
        tile = z_stack[:, start_y:end_y, start_x:end_x]

        # Save the tile as a new TIFF file
        output_file = os.path.join(output_dir, f'tile_{i}_{j}.tiff')
        tiff.imwrite(output_file, tile)

print("Tiles saved successfully.")
