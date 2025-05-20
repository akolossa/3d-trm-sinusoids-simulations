import os
import torch
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
from UNet3D import UNet3D
from config import  IN_CHANNELS, OUT_CHANNELS

tiles = 6
def preprocess_input(tiff_path):
    # Load the TIFF image
    image = tiff.imread(tiff_path)
    
    # Normalize the image to range [0, 1]
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0  # 16-bit normalization
    else:
        image = image.astype(np.float32) / 255.0  # 8-bit normalization
    
    # Add channel and batch dimensions
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Ensure the image has 5 dimensions: [batch_size, channels, depth, height, width]
    if image.ndim == 4:
        image = np.expand_dims(image, axis=2)  # Add depth dimension if missing
    
    return image

def postprocess_output(output_tensor, threshold=0.5):
    # Convert the output tensor to a numpy array
    output_image = output_tensor.cpu().numpy()
    
    # Remove batch and channel dimensions
    output_image = np.squeeze(output_image)
    
    # Apply threshold to get binary image
    output_image = (output_image > threshold).astype(np.uint8) * 255
    
    return output_image

# Load the trained model
DEVICE = torch.device('cpu')  # Force using CPU only-not sure why outofmemory issue
model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
model.load_state_dict(torch.load('/home/arawa/Segmentation_shabaz_FV/output/3Dunet_trained_model_FV3.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def infer_tiff_image(tiff_path):
    # Load and preprocess the input image
    input_image = preprocess_input(tiff_path)
    input_tensor = torch.tensor(input_image, dtype=torch.float32).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities

    # Postprocess the output
    output_image = postprocess_output(output)
    return output_image

# Directory containing the TIFF images
tiff_directory = '/home/arawa/Segmentation_shabaz_FV/3x3_tiles'
output_directory = '/home/arawa/Segmentation_shabaz_FV/inference_3x3tiles'
os.makedirs(output_directory, exist_ok=True)

# Loop through the tiles and perform inference
for i in range(tiles):
    for j in range(tiles):
        tiff_filename = f'tile_{i}_{j}.tiff'
        tiff_path = os.path.join(tiff_directory, tiff_filename)
        if os.path.exists(tiff_path):
            print(f'Processing {tiff_filename}')
            try:
                output = infer_tiff_image(tiff_path)
                
                # Save the output as a TIFF file
                output_tiff_filename = os.path.join(output_directory, f'output_{i}_{j}.tiff')
                tiff.imsave(output_tiff_filename, output)  # Save the output image
            except Exception as e:
                print(f'Error while processing {tiff_filename}: {e}')
        else:
            print(f'{tiff_filename} does not exist.')

print("Inference complete!")