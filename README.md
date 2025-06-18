# 3D Cell Simulations in the Liver

This repository contains the code and resources for my thesis on **3D Cell Simulations in the Liver**, focusing on liver sinusoid segmentation, structure processing, and cellular Potts model (CPM) simulations. The project leverages deep learning for segmentation and GPU-accelerated simulations to study cell movement and interactions within liver sinusoids.

## Repository Structure

- **3D_simulations**: Contains code for running 3D CPM simulations, including:
  - Parameter optimization (grid search and Bayesian methods)
  - Cell movement analysis and comparison with Fernandez-Ruiz et al. (2016) model
  - Coverage time modeling and power-law analysis

- **Expansion_sinusoids**: Tools for processing sinusoidal structures:
  - Diameter adjustment (7-15 µm range)
  - Skeletonization and path extrusion
  - Connected component analysis

- **Segmentation_sinusoids**: 3D U-Net-based segmentation pipeline:
  - Data preparation and augmentation
  - Model training and inference
  - Post-processing (stitching, skeletonization)

## Key Features

### Sinusoid Segmentation
- **3D U-Net architecture** with encoder-decoder structure (64-1024 channels)
- **Data augmentation**:
  - Standard: flips, brightness adjustments, Gaussian noise
  - Enhanced: orientation-selective filtering (Ishikawa et al. 2013)
- **Training**: 300 epochs, AdamW optimizer, BCE loss
- **Post-processing**: Skeletonization, junction detection, volumetric extrusion

### CPM Simulations
- **Parameter optimization**:
  - Grid search over 5 parameters (temperature, actin activity, etc.)
  - Bayesian optimization with Gaussian processes
- **Cell movement analysis**:
  - Volume exploration metrics
  - Velocity and arrest coefficients
- **Comparison** with Fernandez-Ruiz et al. linear model

## Hardware Requirements
- **GPUs**: 4× NVIDIA RTX 2080 Ti (11GB GDDR6 each)
- **CUDA**: Version 12.2
- **Driver**: NVIDIA 535.86.05

## Data Sources
- **Devi et al. 2023**: CD31-labeled mouse liver sinusoids (Zenodo, restricted)
- **Rajakaruna et al. 2020**: Evans Blue/Rhodamine-dextran labeled sinusoids (GitHub)

## Usage

### Segmentation
```bash
cd Segmentation_sinusoids
python train.py --config config.yaml
python infer.py --checkpoint model.pt --input data.tiff
````

### Structure Processing

```bash
cd Expansion_sinusoids
python adjust_diameters.py --input network.mat --output adjusted.npz
```

### Simulations

```bash
cd 3D_simulations
python run_cpm.py --params optimized_params.json
```

## Dependencies

* PyTorch 1.12
* CUDA 12.2
* scikit-optimize (for Bayesian optimization)
* Fiji (for image stitching)

## References

Code builds upon:

* GPU-CPM framework (Sultan et al. 2023)
* No-interaction model (Fernandez-Ruiz et al. 2016)

## Contact

Arawa Kolossa
Email: [arawa@hotmail.it](mailto:arawa@hotmail.it), [arawa.kolossa@ru.nl](mailto:arawa.kolossa@ru.nl)
Radboud University

```

