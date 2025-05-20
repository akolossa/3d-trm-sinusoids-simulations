import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import get_dataloaders
from config import H5_FILE_PATH, VOLUME_DEPTH, BATCH_SIZE

def visualize_all_images(dataset, output_dir, num_images=10):
    os.makedirs(output_dir, exist_ok=True) 
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (inputs, labels) in enumerate(data_loader):
        if i >= num_images:
            break 
        slice_idx = inputs.shape[2] // 2 # Select a slice to visualize (middle slice along depth)
        input_slice = inputs[0, 0, slice_idx, :, :].cpu().numpy() # Get the data from tensors
        label_slice = labels[0, 0, slice_idx, :, :].cpu().numpy()
        input_slice_norm = (input_slice - input_slice.min()) / (input_slice.max() - input_slice.min() + 1e-5) # Normalize images for visualization
        label_slice_norm = (label_slice - label_slice.min()) / (label_slice.max() - label_slice.min() + 1e-5)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Plot and save the images
        axes[0].imshow(input_slice_norm, cmap='gray')
        axes[0].set_title('Input Image')
        axes[1].imshow(label_slice_norm, cmap='gray')
        axes[1].set_title('Label Image')
        plt.savefig(os.path.join(output_dir, f'image_{i}.png'))
        plt.close(fig)
train_loader, val_loader = get_dataloaders(H5_FILE_PATH, VOLUME_DEPTH, BATCH_SIZE) # Load the dataset
visualize_all_images(train_loader.dataset, output_dir='train_images', num_images=10) #  training dataset
visualize_all_images(val_loader.dataset, output_dir='val_images', num_images=10) # val dataset