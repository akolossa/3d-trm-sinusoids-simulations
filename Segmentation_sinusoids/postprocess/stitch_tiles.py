import os
import tifffile as tiff
import numpy as np
from scipy.signal import correlate2d

grid = 6  # it's a 6x6 grid

# Overlap size (in pixels)
overlap = 40

# Initialize a list to store the tiles
tiles_grid = [[None for _ in range(grid)] for _ in range(grid)]

# Loop through the grid positions and load the corresponding tiles
for i in range(grid):  # Rows
    for j in range(grid):  # Columns
        # Construct the filename for each segmented tile
        tile_file_name = f'output_{i}_{j}.tiff'
        tile_path = os.path.join('/home/arawa/Segmentation_shabaz_FV/inference_6x6tiles', tile_file_name)
        
        # Check if the tile file exists
        if not os.path.isfile(tile_path):
            print(f"Segmented tile file {tile_file_name} not found. Skipping.")
            continue
        
        # Load the segmented tile
        tile_image = tiff.imread(tile_path)  # Shape: (Z, H, W)
        
        # Store the tile in the grid
        tiles_grid[i][j] = tile_image

# Verify that all tiles have been loaded
for i in range(grid):
    for j in range(grid):
        if tiles_grid[i][j] is None:
            raise ValueError(f"Tile at position ({i}, {j}) is missing. Cannot proceed with stitching.")

# Get tile dimensions
tile_depth, tile_height, tile_width = tiles_grid[0][0].shape

# Calculate the dimensions of the stitched image
stitched_height = tile_height * grid - overlap * (grid - 1)
stitched_width = tile_width * grid - overlap * (grid - 1)

# Initialize a list to store the stitched Z-slices
stitched_slices = []

# Function to find the best alignment using cross-correlation
def find_best_alignment(tile1, tile2, overlap_size, axis):
    if axis == 0:  # Vertical alignment
        region1 = tile1[-overlap_size:, :]
        region2 = tile2[:overlap_size, :]
    else:  # Horizontal alignment
        region1 = tile1[:, -overlap_size:]
        region2 = tile2[:, :overlap_size]
    
    correlation = correlate2d(region1, region2, mode='valid')
    shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    return shift

# Loop through each Z-slice
for z_index in range(tile_depth):
    stitched_slice = np.zeros((stitched_height, stitched_width), dtype=tiles_grid[0][0].dtype)
    for i in range(grid):
        for j in range(grid):
            # Extract the Z-slice from the current tile
            tile_slice = tiles_grid[i][j][z_index]
            # Calculate the position to place the tile slice in the stitched image
            start_row = i * (tile_height - overlap)
            start_col = j * (tile_width - overlap)
            end_row = start_row + tile_height
            end_col = start_col + tile_width
            
            # Align the tile slice based on similarity
            if i > 0:  # Align with the tile above
                shift = find_best_alignment(tiles_grid[i-1][j][z_index], tile_slice, overlap, axis=0)
                start_row += shift[0]
                end_row = start_row + tile_height
            if j > 0:  # Align with the tile to the left
                shift = find_best_alignment(tiles_grid[i][j-1][z_index], tile_slice, overlap, axis=1)
                start_col += shift[1]
                end_col = start_col + tile_width
            
            # Place the tile slice in the stitched image with logical OR operation
            if i > 0 and j > 0:  # Merge with the tile above and to the left
                stitched_slice[start_row:end_row, start_col:end_col] = np.logical_or(
                    stitched_slice[start_row:end_row, start_col:end_col], tile_slice)
            elif i > 0:  # Merge with the tile above
                stitched_slice[start_row:end_row, start_col:end_col] = np.logical_or(
                    stitched_slice[start_row:end_row, start_col:end_col], tile_slice)
            elif j > 0:  # Merge with the tile to the left
                stitched_slice[start_row:end_row, start_col:end_col] = np.logical_or(
                    stitched_slice[start_row:end_row, start_col:end_col], tile_slice)
            else:  # No merging needed for the first tile
                stitched_slice[start_row:end_row, start_col:end_col] = tile_slice

    stitched_slices.append(stitched_slice)

# Stack all Z-slices to form the full 3D image
stitched_image = np.array(stitched_slices)

# Save the stitched image as a TIFF file
tiff.imwrite('/home/arawa/Segmentation_shabaz_FV/final_image/inference_FI.tiff', stitched_image.astype(np.uint8))

print("Stitched segmented image saved successfully")