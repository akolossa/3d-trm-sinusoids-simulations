import numpy as np
from PIL import Image
import tifffile as tiff
import os

# Directory containing TIFF files
tiff_dir = '/Users/arawa/Downloads/Segmentation_shabaz_FV/partially annotated masks, full image'

# Create directory to save images if it doesn't exist
output_dir = '/Users/arawa/Downloads/Segmentation_shabaz_FV/mask_5x5_tiles'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all TIFF files in the directory
for tiff_file in os.listdir(tiff_dir):
    if tiff_file.endswith('.tif') or tiff_file.endswith('.tiff'):
        # Load the TIFF file
        tiff_path = os.path.join(tiff_dir, tiff_file)
        image = tiff.imread(tiff_path)  # Read the image

        # Extract the file name without extension
        file_name = os.path.splitext(os.path.basename(tiff_file))[0]

        # Check dimensions
        height, width = image.shape  # shape should be (1857, 1857)

        # Define grid size
        grid_size = 5
        tile_height = height // grid_size
        tile_width = width // grid_size

        # Iterate over the grid
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate coordinates for the tiles
                start_x = j * tile_width
                end_x = start_x + tile_width
                start_y = i * tile_height
                end_y = start_y + tile_height

                # Slice the image for the current tile
                tile = image[start_y:end_y, start_x:end_x]

                # Save the tile as a new TIFF file with the original file name and position
                output_file = os.path.join(output_dir, f'{file_name}_tile_{i}_{j}.tiff')
                tiff.imwrite(output_file, tile)

print("Tiles saved successfully.")