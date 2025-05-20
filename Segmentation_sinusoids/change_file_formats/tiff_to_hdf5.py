import h5py
import tifffile as tiff
import numpy as np
import os

tiles_folder = "/home/arawa/Segmentation_shabaz_FV/10x10_tiles"
all_raw_slices = [] #list holding raw slices
all_mask_slices_sinusoids = [] #list holding masks

# Loop through the 10x10 grid of raw images
for i in range(10):  # Rows
    for j in range(10):  # Columns
        # Construct the filename for each tile
        raw_file_name = f"tile_{i}_{j}.tiff"  # Update with the correct naming convention

        # Load the full raw Z-stack image
        try:
            raw_image = tiff.imread(os.path.join(tiles_folder, raw_file_name))  # Load the raw image
        except FileNotFoundError:
            print(f"Warning: {raw_file_name} not found. Skipping this tile.")
            continue

        # Create dictionaries to store masks for quick lookup by Z-index
        masks_dict_sinusoids = {}
        # Loop through possible Z-slice numbers to find corresponding masks
        for slice_num in range(raw_image.shape[0]):  # Assuming all slices are named tile_{i}_{j}_z{slice_num}
            # Construct the sinusoid mask filename
            mask_file_name_sinusoids = f"tile_{i}_{j}_z{slice_num}.tif"  # Sinusoid mask
            # Attempt to load the sinusoid mask image
            try:
                mask_image_sinusoids = tiff.imread(os.path.join(tiles_folder, mask_file_name_sinusoids))
                masks_dict_sinusoids[slice_num] = mask_image_sinusoids
            except FileNotFoundError:
                print(f"Warning: Sinusoid mask file {mask_file_name_sinusoids} not found. Skipping.")
        # Loop through all Z-slices of the raw image
        for z_index in range(raw_image.shape[0]):  # Loop through each Z-slice
            # Extract the corresponding Z-slice from the raw 3D volume
            raw_slice = raw_image[z_index]  # Get the specific Z-slice
            # Check if there is a corresponding sinusoid mask for the current Z-index
            if z_index in masks_dict_sinusoids:
                mask_image_sinusoids = masks_dict_sinusoids[z_index]
                # Print the shapes for debugging
                print(f"Tile: {raw_file_name}, Z-index: {z_index}, Raw slice shape: {raw_slice.shape}, Sinusoid mask shape: {mask_image_sinusoids.shape}")
                # Check if shapes match
                if raw_slice.shape != mask_image_sinusoids.shape:
                    # Resize the mask to match the raw slice
                    print(f"Warning: Shape mismatch for Z-index {z_index}. Resizing sinusoid mask to match raw slice.")
                    mask_image_sinusoids = np.resize(mask_image_sinusoids, raw_slice.shape)   
                # Append to lists
                all_raw_slices.append(raw_slice)
                all_mask_slices_sinusoids.append(mask_image_sinusoids)
            else:
                print(f"No sinusoid mask available for Z-index: {z_index} in tile {raw_file_name}. Skipping this slice.")

# Convert lists to NumPy arrays
all_raw_slices = np.array(all_raw_slices)
all_mask_slices_sinusoids = np.array(all_mask_slices_sinusoids)

# Create an HDF5 file for training
with h5py.File('/home/arawa/Segmentation_shabaz_FV/3D_unet/training_data.h5', 'w') as f:
    f.create_dataset('raw', data=all_raw_slices, compression="gzip")
    f.create_dataset('sinusoids', data=all_mask_slices_sinusoids, compression="gzip")
print("HDF5 file created successfully with all available Z-slices and masks.")

# Check content of HDF5 file
with h5py.File('/home/arawa/Segmentation_shabaz_FV/3D_unet/training_data.h5', 'r') as f:
    print("Datasets in the file:")  # Print the keys (datasets) in the file
    for key in f.keys():
        print(f" - {key}")
    raw_data = f['raw'][:]  # Load all raw data
    sinusoids_data = f['sinusoids'][:]  # load sinusoid mask
    print("\nRaw data shape:", raw_data.shape)
    print("Sinusoids data shape:", sinusoids_data.shape)
