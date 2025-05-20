import os
from pathlib import Path

class Config:
    BASE_DATA_DIR = Path("/media/datadrive/arawa/expansion_sinusoids") # Base directory for all non-Python files (data, outputs, etc.)
    SINUSOID_PATHS = {
        "segmented": Path("/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz"),
        "paper": Path("/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}.npz")
    }
    # Output direct
    PREPROCESSED_DIR = BASE_DATA_DIR / "preprocessed"
    TRAIN_DATA_DIR = BASE_DATA_DIR / "train_data"
    OUTPUTS_DIR = BASE_DATA_DIR / "outputs"
    VISUALIZATIONS_DIR = BASE_DATA_DIR / "visualizations"
    # parameters
    DIMENSION = 128  # input size
    TARGET_DIMENSION = 1028  # output size
    SEED_SIZE = 64  # Size central seed region to keep
    MASK_WIDTH = 32  # Width of the masked boundary region
    BATCH_SIZE = 2  
    NUM_WORKERS = 4  
    TIMESTEPS = 1000  # diffusion timesteps
    SCHEDULE = "linear"  # Noise schedule type
    
    def __init__(self): #create directories
        os.makedirs(self.PREPROCESSED_DIR, exist_ok=True)
        os.makedirs(self.TRAIN_DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(self.VISUALIZATIONS_DIR, exist_ok=True)

cfg = Config() # Instantiate the config
