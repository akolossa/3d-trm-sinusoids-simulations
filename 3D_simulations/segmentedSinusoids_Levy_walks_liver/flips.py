import scipy.io
import numpy as np
import tifffile as tiff
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load the .mat file
mat_file = '/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/IMG_OPENED_3D.mat'
data = scipy.io.loadmat(mat_file)

key_of_interest = 'IMG_OPENED_3D_Evans_old'  

if key_of_interest in data:
    variable_data = data[key_of_interest]
    
    # Check if the variable is 3D
    if len(variable_data.shape) == 3:
        print(f"The variable '{key_of_interest}' is 3D with shape: {variable_data.shape}")

        # Reorder axes to (z, x, y) bc it's saved as (x, y, z)
        reordered_data = np.transpose(variable_data, (2, 0, 1))  # Convert (x, y, z) -> (z, x, y)
        original_shape = reordered_data.shape
        print(f"Reordered shape: {original_shape}")

        # Save the original data
        original_tiff_file = f'/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/zxyReordered_IMG_3D_yx{original_shape[1]}_z{original_shape[0]}.tiff'
        tiff.imwrite(original_tiff_file, reordered_data.astype(np.float32))  # Save as 32-bit TIFF
        npz_file = f'/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/zxyReordered_IMG_3D_yx{original_shape[1]}_z{original_shape[0]}.npz'
        np.savez_compressed(npz_file, data=reordered_data)
        print(f"Original 3D data successfully saved as '{original_tiff_file}' and '{npz_file}'")

        # Number of times to repeat
        repeat_times = 13
        
        # Create expanded array with alternating flips
        expanded_data = np.zeros((original_shape[0] * repeat_times, original_shape[1], original_shape[2]), dtype=np.float32)
        for i in range(repeat_times):
            if i % 2 == 0:
                expanded_data[i * original_shape[0]:(i + 1) * original_shape[0]] = reordered_data
            else:
                expanded_data[i * original_shape[0]:(i + 1) * original_shape[0]] = reordered_data[::-1, ::-1, :]  # Flip along z and x

        expanded_shape = expanded_data.shape
        print(f"Expanded shape: {expanded_shape}")

        # Save the expanded data
        expanded_tiff_file = f'/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_zxyReordered_IMG_OPENED_3D_{expanded_shape[1]}x{expanded_shape[2]}_z{expanded_shape[0]}.tiff'
        tiff.imwrite(expanded_tiff_file, expanded_data.astype(np.float32))  # Save as 32-bit TIFF
        expanded_npz_file = f'/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_zxyReordered_IMG_OPENED_3D_{expanded_shape[1]}x{expanded_shape[2]}_z{expanded_shape[0]}.npz'
        np.savez_compressed(expanded_npz_file, data=expanded_data)
        print(f"Expanded 3D data successfully saved as '{expanded_tiff_file}' and '{expanded_npz_file}'")

    else:
        print(f"The variable '{key_of_interest}' is not 3D. Its shape is: {variable_data.shape}")
else:
    print(f"The variable '{key_of_interest}' was not found in the .mat file.")
