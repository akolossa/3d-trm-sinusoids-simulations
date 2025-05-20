import torch.nn as nn

# Paths
H5_FILE_PATH = '/media/datadrive/arawa/Segmentation_shabaz_FV/Segmentation_shabaz_FV.2/3D_unet/training_data.h5'
MODEL_SAVE_PATH = 'liver-3d-simulations/Segmentation_sinusoids/trained_models/model_NO_objSel_filter.pth'
LOG_DIR = '/home/arawa/liver-3d-simulations/Segmentation_sinusoids/logs'

# Hyperparameters
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3
VOLUME_DEPTH = 4
BATCH_SIZE = 8
LOSS_FUNCTION = nn.BCEWithLogitsLoss()