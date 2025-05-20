
#this implementation is inspired by the methods described in Ishikawa et al. 2013
import numpy as np
import cv2
import h5py
import os
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel

def os_filter(image):
    """
    Apply an orientation-selective filter to enhance sinusoid boundaries.
    This helps emphasize structures that are important for the model to learn.
    """
    smoothed = gaussian_filter(image, sigma=(1, 2))  # Approximate directional smoothing
    gradient_x = np.gradient(smoothed, axis=1)
    gradient_y = np.gradient(smoothed, axis=0)
    
    # Combine gradients to enhance sinusoid boundaries
    os_filtered_image = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return os_filtered_image

def preprocess_image(image, enhance_edges=False):
    """
    Preprocess a single image: Normalize and optionally apply edge enhancement.
    Parameters:
    - enhance_edges: If True, apply an orientation-selective filter.
    """
    # Optional: Apply edge enhancement to highlight sinusoids
    if enhance_edges:
        image = os_filter(image)
    else:
        # Apply Sobel filter if you need a simpler alternative
        image = sobel(image)

    # Normalize the image
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-7)  # Normalize to range [0, 1]

    return image

def preprocess_dataset(h5_file_path, output_dir, enhance_edges=False):
    """
    Preprocess the dataset for segmentation model training.
    Save the preprocessed images and their masks in the output directory.
    Parameters:
    - enhance_edges: If True, apply an orientation-selective filter for edge enhancement.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_file_path, 'r') as f:
        raw_images = f['raw'][:]  # Shape: (num_slices, height, width)
        sinusoids = f['sinusoids'][:]  # Shape: (num_slices, height, width)

    preprocessed_images = []
    preprocessed_masks = []

    for i, image in enumerate(raw_images):
        # Preprocess the image (normalize and optionally enhance edges)
        preprocessed_image = preprocess_image(image, enhance_edges=enhance_edges)
        preprocessed_images.append(preprocessed_image)
        
        # Use the corresponding sinusoid mask
        mask = sinusoids[i]
        preprocessed_masks.append(mask)
        
        # Save the preprocessed image and its mask
        image_output_path = os.path.join(output_dir, f'preprocessed_image_{i}.png')
        mask_output_path = os.path.join(output_dir, f'preprocessed_mask_{i}.png')
        cv2.imwrite(image_output_path, img_as_ubyte(preprocessed_image))
        cv2.imwrite(mask_output_path, img_as_ubyte(mask))

    return np.array(preprocessed_images), np.array(preprocessed_masks)

# Example usage
h5_file_path = '/media/datadrive/arawa/Segmentation_shabaz_FV/Segmentation_shabaz_FV.2/3D_unet/training_data.h5'
output_dir = '/media/datadrive/arawa/Segmentation_shabaz_FV/output_preprocess'
preprocessed_images, preprocessed_masks = preprocess_dataset(h5_file_path, output_dir, enhance_edges=True)
print(f"Preprocessed images shape: {preprocessed_images.shape}")