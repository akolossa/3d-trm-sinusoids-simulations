import torch
from UNet3D import UNet3D  
from torchviz import make_dot

model = UNet3D(in_channels=1, out_channels=1)  # Example input/output channels (you can adjust them)
dummy_input = torch.randn(8, 1, 4, 185, 185)  # Batch size 8, 1 input channel, 4x185x185 volume

# Export the model to an ONNX file
torch.onnx.export(model, dummy_input, "unet3d.onnx", opset_version=11, input_names=['input'], output_names=['output']) #use Netron for diagram


output = model(dummy_input) # Perform a forward pass to get the model output

dot = make_dot(output, params=dict(model.named_parameters())) # Generate the graph using torchviz
dot.render('unet3d_graph', format='png')


