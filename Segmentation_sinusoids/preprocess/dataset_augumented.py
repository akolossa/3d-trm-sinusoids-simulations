from sympy import degree
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import torchio as tio
import cv2
import os
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
#from skimage.filters import sobel
from preprocess.preprocess import os_filter, preprocess_image

# Updating the LiverVolumeDataset
def generate_transformed_images(subject):
    transformed_subjects = []
    
    # 1. Flip Transformations
    flip = tio.RandomFlip(axes=('LR', 'AP', 'SI'), flip_probability=1.0)  # Flip along all axes
    transformed_subjects.append(flip(subject))

    # Rotation every x degrees
    #for angle in range(10, 360, 10):  #first number starts, last number adds till middle number is reached
            #transform = tio.RandomAffine(degrees=(0, angle, 0),  # Rotate around the z-axis for each slice
                                            #scales=(1, 1, 1),    # No scaling - still applies it idk why
                                            #center='image',      # Rotate around the image center
                                            #default_pad_value=0, # Use black padding
            #transformed_subjects.append(transform(subject))
    
    # 2. Brightness Transformations
    brightness_values = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    for brightness in brightness_values:
        log_brightness = np.log(brightness)
        transform = tio.RandomGamma(log_gamma=(log_brightness, log_brightness))
        transformed_subjects.append(transform(subject))
    
    # 3. Noise Transformations
    noise_levels = [0.01, 0.03, 0.05, 0.07]
    for noise in noise_levels:
        transform = tio.RandomNoise(std=noise)
        transformed_subjects.append(transform(subject))

    # 4. Preprocessed Transformations using preprocess.py
    # Extract raw volume to apply preprocessing
    # raw_volume = subject['raw'].data.numpy()[0]  # Shape: (depth, height, width)
    # for enhance_edges in [True, False]:
    #     preprocessed_volume = np.stack([preprocess_image(slice, enhance_edges=enhance_edges) for slice in raw_volume])
    #     preprocessed_tensor = torch.tensor(preprocessed_volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, depth, height, width)
        
    #     # Ensure dimensions match
    #     assert preprocessed_tensor.shape == subject['raw'].data.shape, "Preprocessed image dimensions do not match original dimensions"
        
    #     preprocessed_subject = tio.Subject(
    #         raw=tio.ScalarImage(tensor=preprocessed_tensor),
    #         label=subject['label']  # Use the same label as the original
    #     )
    #     transformed_subjects.append(preprocessed_subject)
    
    # Ensure all transformed subjects maintain their dimensions
    for transformed in transformed_subjects:
        assert transformed['raw'].data.shape == subject['raw'].data.shape, "Transformed image dimensions do not match original dimensions"
        assert transformed['label'].data.shape == subject['label'].data.shape, "Transformed label dimensions do not match original dimensions"

    return transformed_subjects

class LiverVolumeDataset(Dataset):
    def __init__(self, h5_file_path, volume_depth=4):
        self.volume_depth = volume_depth  # Number of slices per volume
        self.data = []
        with h5py.File(h5_file_path, 'r') as f:  # Load the HDF5 dataset
            self.raw = f['raw'][:]  # Shape: (num_slices, height, width)
            self.sinusoids = f['sinusoids'][:]  # Shape: (num_slices, height, width)
        
        self.num_volumes = len(self.raw) // self.volume_depth  # Calculate the number of volumes
        for idx in range(self.num_volumes):  # Prepare the data
            # Extract a sequence of slices to form a 3D volume
            start_idx = idx * self.volume_depth
            end_idx = start_idx + self.volume_depth
            raw_volume = self.raw[start_idx:end_idx]  # Shape: (depth, height, width)
            sinusoids_volume = self.sinusoids[start_idx:end_idx]
            
            if raw_volume.dtype == np.uint16:  # Normalize the raw volume to range [0, 1]
                raw_volume = raw_volume.astype(np.float32) / 65535.0  # 16-bit normalization
            else:
                raw_volume = raw_volume.astype(np.float32) / 255.0  # 8-bit normalization
            
            # Create a combined label volume
            combined_labels = np.zeros_like(sinusoids_volume, dtype=np.uint8)  # Initialize as 0 (background)
            combined_labels[sinusoids_volume > 0] = 1  # Binary mask
            
            # Convert to torch tensors and add channel dimension
            raw_tensor = torch.tensor(raw_volume, dtype=torch.float32).unsqueeze(0)  # Shape: (1, depth, height, width)
            labels_tensor = torch.tensor(combined_labels, dtype=torch.float32).unsqueeze(0)  # Shape: (1, depth, height, width)
            
            subject = tio.Subject(  # Create a torchio Subject
                raw=tio.ScalarImage(tensor=raw_tensor),
                label=tio.LabelMap(tensor=labels_tensor)
            )
            
            transformed_subjects = generate_transformed_images(subject)  # Generate transformed subjects
            transformed_subjects.append(subject)  # Add the original subject
            
            for transformed in transformed_subjects:
                raw_transformed = transformed['raw'].data
                label_transformed = transformed['label'].data
                self.data.append((raw_transformed, label_transformed))
        
        num_original = self.num_volumes
        num_transformed = len(self.data) - num_original
        print(f"Number of original images: {num_original}")
        print(f"Number of transformed images: {num_transformed}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        raw_tensor, labels_tensor = self.data[idx]
        return raw_tensor, labels_tensor
