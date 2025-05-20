import vtk
from vtk.util import numpy_support
import numpy as np
import os
from scipy.ndimage import binary_dilation

def create_sphere_mask(shape, center, radius):
    """Create a binary mask with a sphere of given radius centered at the given point."""
    grid = np.indices(shape).transpose(1, 2, 3, 0)
    distances = np.linalg.norm(grid - np.array(center), axis=-1)
    mask = distances <= radius
    return mask

def create_rectangular_mask(shape, center, dimensions):
    """Create a binary mask with a rectangular shape centered at the given point."""
    mask = np.zeros(shape, dtype=bool)
    z, x, y = center
    dz, dx, dy = dimensions
    z_min, z_max = max(0, z - dz // 2), min(shape[0], z + dz // 2 + 1)
    x_min, x_max = max(0, x - dx // 2), min(shape[1], x + dx // 2 + 1)
    y_min, y_max = max(0, y - dy // 2), min(shape[2], y + dy // 2 + 1)
    mask[z_min:z_max, x_min:x_max, y_min:y_max] = True
    return mask

def save_vti(cell_type, centroids, timestep, dimension, version, runs, n_cells, exploration=False, max_distance=14, activity=None):
    # Create separate arrays for each component
    t_cells = (cell_type == 1).astype(np.uint8)  # T-cell volume
    explored_cells = exploration.astype(np.uint8)  # Exploration volume (binary)
    background_cells = (cell_type == 2).astype(np.uint8)
    sinusoid_cells = (cell_type == 0).astype(np.uint8)  # Sinusoid volume
    activity = activity.astype(np.uint16)  # Energy volume, somehow visualises the sinusoids
    t_cell_activity = activity * t_cells
    # Create a mask for the regions around the centroids
    mask = np.zeros(cell_type.shape, dtype=bool)
    for centroid in centroids:
        # Adjust centroid coordinates from (x, y, z) to (z, x, y)
        adjusted_centroid = [centroid[2], centroid[1], centroid[0]]
        # Ensure centroid coordinates are integers and within the grid bounds
        adjusted_centroid = np.round(adjusted_centroid).astype(int)
        if np.all(adjusted_centroid >= 0) and np.all(adjusted_centroid < np.array(cell_type.shape)):
            mask |= create_rectangular_mask(cell_type.shape, adjusted_centroid, (50, 50, 10))

    # Find background cells within the mask
    background_nearby = (background_cells & mask).astype(np.uint8)
    sinusoids_nearby = (sinusoid_cells & mask).astype(np.uint8)
    # transform sinusoids into borders
    dilated_sinusoids_nearby = binary_dilation(sinusoids_nearby)
    #border_sinusoids_nearby = dilated_sinusoids_nearby & ~sinusoids_nearby

    # Convert numpy arrays to VTK arrays
    t_cells_vtk = numpy_support.numpy_to_vtk(t_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    explored_cells_vtk = numpy_support.numpy_to_vtk(explored_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    background_cells_vtk = numpy_support.numpy_to_vtk(background_nearby.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    #border_sinusoids_nearby_vtk = numpy_support.numpy_to_vtk(border_sinusoids_nearby.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    activity_vtk = numpy_support.numpy_to_vtk(activity.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    t_cells_activity_vtk = numpy_support.numpy_to_vtk(t_cell_activity.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    # Create VTK image data
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(cell_type.shape)

    # Add the T-cells, exploration data, and background cells as separate volumes (arrays)
    image_data.GetPointData().AddArray(t_cells_vtk)
    image_data.GetPointData().AddArray(explored_cells_vtk)
    image_data.GetPointData().AddArray(background_cells_vtk)
    #image_data.GetPointData().AddArray(border_sinusoids_nearby_vtk)
    image_data.GetPointData().AddArray(activity_vtk)
    image_data.GetPointData().AddArray(t_cells_activity_vtk)
    # Set names for each array so they can be identified in ParaView
    t_cells_vtk.SetName("T-Cells")
    explored_cells_vtk.SetName("Explored-Volume")
    background_cells_vtk.SetName("Background-Cells near T-Cells")
    #border_sinusoids_nearby_vtk.SetName("Border-Sinusoids near T-Cells")
    activity_vtk.SetName("Activity")
    t_cells_activity_vtk.SetName("T-Cell Activity")
    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    vti_dir = f"/home/arawa/Shabaz_simulation/figure_7_ak/current_vti_files/output_{version}_dim{dimension}_{n_cells}cells_r{runs}"
    os.makedirs(vti_dir, exist_ok=True)
    writer.SetFileName(os.path.join(vti_dir, f'timestep_{timestep}.vti'))
    writer.SetInputData(image_data)
    writer.Write()

    print(f"VTI file saved for timestep {timestep} at {vti_dir}")