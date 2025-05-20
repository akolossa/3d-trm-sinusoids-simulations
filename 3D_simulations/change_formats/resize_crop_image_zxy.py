import numpy as np
import scipy.ndimage as nd

def crop_to_match_dimension(image, target_dimension):
    """
    Crop the z, y, and x dimensions of the image to match the target dimension without altering the inside of the image.
    
    Parameters:
    - image: The input 3D numpy array.
    - target_dimension: The target dimension for z, y, and x.
    
    Returns:
    - Cropped image with dimensions (target_dimension, target_dimension, target_dimension).
    """
    z_dim, y_dim, x_dim = image.shape

    # Calculate the cropping indices for z dimension
    z_start = (z_dim - target_dimension) // 2
    z_end = z_start + target_dimension

    # Calculate the cropping indices for y dimension
    y_start = (y_dim - target_dimension) // 2
    y_end = y_start + target_dimension

    # Calculate the cropping indices for x dimension
    x_start = (x_dim - target_dimension) // 2
    x_end = x_start + target_dimension

    # Crop the image
    cropped_image = image[z_start:z_end, y_start:y_end, x_start:x_end]

    return cropped_image

def main():
    dimension = 512  # Target dimension for z, y, and x
    # input_path = "/home/arawa/Shabaz_simulation/segmentedSinusoids_AK_FV.npz"
    # output_path = f"/home/arawa/Shabaz_simulation/segmentedSinusoids_AK_FV_cropped{dimension}.npz"
    #paper image
    input_path = "/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_zxyReordered_IMG_OPENED_3D_512x512_z546.npz"
    output_path = f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}.npz"
    # Load the image
    print("Loading image...")
    image = np.load(input_path)["data"]


    # Crop the image
    print("Cropping image to match the target dimension...")
    cropped_image = crop_to_match_dimension(image, dimension)

    # Save the cropped image in .npz format
    print(f"Saving cropped image to {output_path}...")
    np.savez_compressed(output_path, image=cropped_image)
    print("Cropped image saved successfully.")

if __name__ == "__main__":
    main()