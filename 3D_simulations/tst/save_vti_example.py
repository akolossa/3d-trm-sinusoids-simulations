import vtk
import numpy as np
import os
from vtk.util import numpy_support

def save_vti(cell_type, i, dimension, version):
    # Create separate arrays for each component
    t_cells = (cell_type == 1).astype(np.uint8) # shows t-cells 
    b_cells = (cell_type == 0).astype(np.uint8) # shows b-cells

    # Convert numpy arrays to VTK arrays
    t_cells_vtk = numpy_support.numpy_to_vtk(t_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    b_cells_vtk = numpy_support.numpy_to_vtk(b_cells.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create VTK image data
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(cell_type.shape)
    image_data.GetPointData().AddArray(t_cells_vtk)
    image_data.GetPointData().AddArray(b_cells_vtk)

    # Set array names for visualization in ParaView
    t_cells_vtk.SetName("T-Cells")
    b_cells_vtk.SetName("B-Cells")

    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    vti_dir = f"example3d_output_{dimension}_{version}"
    os.makedirs(vti_dir, exist_ok=True)
    writer.SetFileName(os.path.join(vti_dir, f'timestep_{i}.vti'))
    writer.SetInputData(image_data)
    writer.Write()
    print(f"VTI file saved in directory: {vti_dir}")