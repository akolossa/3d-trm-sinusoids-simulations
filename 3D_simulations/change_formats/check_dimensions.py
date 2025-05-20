import numpy as np

# Load the NPZ file
data = np.load('/media/datadrive/arawa/Shabaz_simulation_22_04_25/segmentedSinusoids_npz/segmentedSinusoids_AK_FV.npz')
data = np.load('/media/datadrive/arawa/Shabaz_simulation_22_04_25/figure_7_ak/segmentedSinusoids_Levy_walks_liver/zxyReordered_IMG_3D_yx512_z42.npz')
data = np.load('/media/datadrive/arawa/Shabaz_simulation_22_04_25/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_zxyReordered_IMG_OPENED_3D_512x512_z546.npz')
# Print the dimensions of each array in the NPZ file
for key, value in data.items():
    print(f"Dimension of {key}: {value.shape}")
