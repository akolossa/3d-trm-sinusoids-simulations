import vtk
from vtk.util import numpy_support
import numpy as np
import os
from scipy.ndimage import binary_dilation

# Load the segmented liver image
dimension = 768  # Adjust the dimension as needed
version = 'MySinusoids_adjusted'
liver = np.load(f"/home/arawa/Shabaz_simulation/segmentedSinusoids_npz/segmentedSinusoids_AK_FV_cropped{dimension}_new_adjusted.npz")["image"]
# version = 'PaperSinusoid_new'
# liver = np.load(f"/home/arawa/Shabaz_simulation/figure_7_ak/segmentedSinusoids_Levy_walks_liver/flipped_IMG_3D_cropped{dimension}_new.npz")["image"] #loads sinusoids from paper

# Perform binary dilation and compute the border cells
sinusoid_cells = (liver == 0).astype(np.uint8)  # Sinusoids are labeled as 0
dilated_sinusoids = binary_dilation(sinusoid_cells)
border_cells = dilated_sinusoids & ~sinusoid_cells

# Compute the background cells (everything that's not a sinusoid)
background_cells = (liver != 0).astype(np.uint8)

# Convert numpy arrays to VTK arrays
sinusoid_cells_vtk = numpy_support.numpy_to_vtk(sinusoid_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
border_cells_vtk = numpy_support.numpy_to_vtk(border_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
background_cells_vtk = numpy_support.numpy_to_vtk(background_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

# Create VTK image data
image_data = vtk.vtkImageData()
image_data.SetDimensions(liver.shape)

# Add the sinusoid cells, border cells, and background cells as separate volumes (arrays)
image_data.GetPointData().AddArray(sinusoid_cells_vtk)
image_data.GetPointData().AddArray(border_cells_vtk)
image_data.GetPointData().AddArray(background_cells_vtk)

# Set names for each array so they can be identified in ParaView
sinusoid_cells_vtk.SetName("Background")
border_cells_vtk.SetName("Sinusoid-Borders")
background_cells_vtk.SetName("Sinusoids")

# Write VTI file
writer = vtk.vtkXMLImageDataWriter()
vti_dir = "/home/arawa/Shabaz_simulation/figure_7_ak"
os.makedirs(vti_dir, exist_ok=True)
writer.SetFileName(os.path.join(vti_dir, f'{version}{dimension}.vti'))
writer.SetInputData(image_data)
writer.Write()

print(f"VTI file saved at {vti_dir}")