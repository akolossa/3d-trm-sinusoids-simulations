import numpy as np
import os
from scipy.ndimage import label, find_objects
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dimension =128
dimension =768
#liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}.npz")["image"] 
#liver = np.load(f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}.npz")["image"] #loads sinusoids from paper
#liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")["image"] 
#liver = np.load(f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}_new_adjusted.npz")["image"] #loads sinusoids from paper
liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")["image"] 

# Perform connected component analysis
structure = np.ones((3, 3, 3), dtype=int)  # Define connectivity for 3D
labeled_array, num_features = label(liver, structure)
label_pos = (labeled_array != 0).nonzero()
liver_indices = (liver != 0).nonzero()

# Find the sizes of the connected components
component_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude the background component

# Sort the component sizes in descending order
sorted_indices = np.argsort(component_sizes)[::-1]
sorted_sizes = component_sizes[sorted_indices]

# Print the number of connected components and the sizes of the largest ones
print(f"Number of connected components: {num_features}")
print(f"Sizes of the largest connected components: {sorted_sizes[:10]}")  # Print the sizes of the 10 largest components

#voxel plot labeled array
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
ax.scatter(liver_indices[0], liver_indices[1], liver_indices[2], c="blue", alpha=0.2, s=0.1)   
#fig.savefig(f"/home/arawa/Shabaz_simulation/jan_fig")

# #SAVE LARGEST COMPONENT
# # Identify the largest connected component
# largest_component_label = np.argmax(component_sizes) + 1  # +1 because bincount excludes the background

# # Create a mask for the largest connected component
# largest_component_mask = (labeled_array == largest_component_label)

# # Save the largest connected component to a new .npz file
# output_file = f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new.npz"
# #output_file = f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}_new.npz"
# np.savez_compressed(output_file, image=largest_component_mask)

# print(f"Largest connected component saved to {output_file}")
