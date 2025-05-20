import numpy as np
import h5py
from parameters_config import parameters

def npz_to_h5(npz_file, h5_file):
    # Load the .npz file
    data = np.load(npz_file)
    
    # Create a new .h5 file
    with h5py.File(h5_file, 'w') as f:
        # Iterate over each item in the .npz file and save it to the .h5 file
        for key in data:
            f.create_dataset(key, data=data[key], compression="gzip")
    
    print(f"Data from {npz_file} saved to {h5_file}")

if __name__ == "__main__":
    dimension = parameters['dimension']
    version = parameters['version']
    
    npz_file = f"/home/arawa/Shabaz_simulation/segmentedSinusoids_AK_FV_cropped{dimension}.npz"  # Input .npz file
    h5_file = f"/home/arawa/Shabaz_simulation/sim_vid_output_{dimension}_{version}/sinusoids_structure_4d_{dimension}_{version}.h5"  # Output .h5 file

    npz_to_h5(npz_file, h5_file)