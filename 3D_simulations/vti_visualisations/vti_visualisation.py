import vtk
from vtk.util import numpy_support
import numpy as np
import os 
from scipy.ndimage import binary_dilation

#this increases computational time but it works
def chebyshev_distance_transform(input_array, shape):
    grid = np.indices(shape).reshape(3, -1).T # Create a 3D grid of indices
    t_cell_indices = np.array(np.where(input_array == 1)).T # Get the indices of the T-cells (where cell_type == 1)
    chebyshev_distance = np.full(input_array.shape, np.inf) # Calculate the Chebyshev distance from each voxel to the nearest T-cell
    for idx in grid: # Compute the Chebyshev distance to the closest T-cell      
        distances = np.max(np.abs(t_cell_indices - idx), axis=1)
        chebyshev_distance[idx[0], idx[1], idx[2]] = np.min(distances)

    return chebyshev_distance



def save_vti(cell_type, timestep, dimension, version, runs, n_cells, exploration=False, max_distance=7):
    # Create separate arrays for each component
    t_cells = (cell_type == 1).astype(np.uint8)  # T-cell volume
    explored_cells = exploration.astype(np.uint8)  # Exploration volume (binary)
    # sinusoid_cells = (cell_type == 0).astype(np.uint8)  # Sinusoid volume
    background_cells = (cell_type == 2).astype(np.uint8)  
    distance_from_t_cells = chebyshev_distance_transform(t_cells, cell_type.shape)

    # Find sinusoid cells within max_distance from T-cells
    # sinusoid_cells_nearby = (sinusoid_cells & (distance_from_t_cells <= max_distance)).astype(np.uint8)
    background_nearby = (background_cells & (distance_from_t_cells <= max_distance)).astype(np.uint8)
    # transform sinusoids into borders
    # dilated_sinusoids_nearby = binary_dilation(sinusoid_cells_nearby)
    # border_cells_nearby = dilated_sinusoids_nearby & ~sinusoid_cells_nearby
    # dilated_sinusoids = binary_dilation(sinusoid_cells)
    # border_cells = dilated_sinusoids & ~sinusoid_cells
    # convert numpy arrays to VTK arrays
    t_cells_vtk = numpy_support.numpy_to_vtk(t_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    explored_cells_vtk = numpy_support.numpy_to_vtk(explored_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    background_cells_vtk = numpy_support.numpy_to_vtk(background_nearby.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    #sinusoid_cells_vtk = numpy_support.numpy_to_vtk(sinusoid_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    #sinusoid_cells_nearby_vtk = numpy_support.numpy_to_vtk(sinusoid_cells_nearby.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    # border_cells_nearby_vtk = numpy_support.numpy_to_vtk(border_cells_nearby.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    # border_cells_vtk = numpy_support.numpy_to_vtk(border_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    # Create VTK image data
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(cell_type.shape)

    # Add the T-cells, exploration data, sinusoid cells, and nearby sinusoid cells as separate volumes (arrays)
    image_data.GetPointData().AddArray(t_cells_vtk)
    image_data.GetPointData().AddArray(explored_cells_vtk)
    image_data.GetPointData().AddArray(background_cells_vtk)
    #image_data.GetPointData().AddArray(sinusoid_cells_vtk)
    #image_data.GetPointData().AddArray(sinusoid_cells_nearby_vtk)
    #image_data.GetPointData().AddArray(border_cells_nearby_vtk)
    #image_data.GetPointData().AddArray(border_cells_vtk)

    # Set names for each array so they can be identified in ParaView
    t_cells_vtk.SetName("T-Cells")
    explored_cells_vtk.SetName("Explored-Volume")
    background_cells_vtk.SetName("Background-Cells near T-Cells")
    #sinusoid_cells_vtk.SetName("Sinusoids")
    #sinusoid_cells_nearby_vtk.SetName("Sinusoids-Near-T-Cells")
    #border_cells_vtk.SetName("Sinusoid-Borders")
    #border_cells_nearby_vtk.SetName("Sinusoid-Borders-Nearby")

    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    vti_dir = f"/home/arawa/Shabaz_simulation/figure_7_ak/current_vti_files/output_{version}_dim{dimension}_{n_cells}cells_r{runs}" 
    os.makedirs(vti_dir, exist_ok=True)
    writer.SetFileName(os.path.join(vti_dir, f'timestep_{timestep}.vti'))
    writer.SetInputData(image_data)
    writer.Write()

    print(f"VTI file saved for timestep {timestep} at {vti_dir}")

