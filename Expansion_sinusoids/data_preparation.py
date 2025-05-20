import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Dict
import os
from config import cfg
from scipy.ndimage import rotate, zoom
import random
import matplotlib.pyplot as plt

def load_sinusoid_data() -> Dict[str, np.ndarray]:
    """Load both sinusoid datasets from NPZ files."""
    data = {}
    for name, path_template in cfg.SINUSOID_PATHS.items():
        path = str(path_template).format(dimension=cfg.DIMENSION)
        print(f"Loading data from: {path}")
        data[name] = np.load(path)["image"]
        assert data[name].shape == (cfg.DIMENSION, cfg.DIMENSION, cfg.DIMENSION), \
            f"Expected shape {(cfg.DIMENSION,)*3}, got {data[name].shape}"
    return data

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 1] range."""
    volume = volume.astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    return volume

def create_mask(central_only: bool = False, oblique_prob: float = 0.5) -> np.ndarray:
    """
    Create a 3D mask with options for:
    - Central rectangular mask
    - Randomly positioned rectangular mask
    - Oblique (parallelepiped) mask
    Args:
        central_only: If True, forces central positioning
        oblique_prob: Probability (0-1) of generating oblique mask when not central
    Returns:
        3D numpy array with mask values (0 or 1)
    """
    mask = np.zeros((cfg.DIMENSION, cfg.DIMENSION, cfg.DIMENSION), dtype=np.float32)
    
    # Determine mask position
    if central_only:
        start = (cfg.DIMENSION - cfg.SEED_SIZE) // 2
        coords = [slice(start, start + cfg.SEED_SIZE)] * 3
        use_oblique = False  # Never use oblique for central masks
    else:
        max_offset = cfg.DIMENSION - cfg.SEED_SIZE
        starts = [np.random.randint(0, max_offset + 1) if max_offset > 0 else 0 for _ in range(3)]
        coords = [slice(s, s + cfg.SEED_SIZE) for s in starts]
        use_oblique = random.random() < oblique_prob

    if use_oblique:
        # Create oblique parallelepiped mask
        z, y, x = np.ogrid[coords[0], coords[1], coords[2]]
        
        # Apply shear transformations (15-30 degree max angle)
        shear_x = np.tan(np.radians(random.uniform(-25, 25)))
        shear_y = np.tan(np.radians(random.uniform(-25, 25)))
        shear_z = np.tan(np.radians(random.uniform(-25, 25)))
        
        # Transform coordinates
        x = x.astype(float) + shear_y * (y - y.mean()) + shear_z * (z - z.mean())
        y = y.astype(float) + shear_x * (z - z.mean())
        
        # Convert back to integer indices
        x = np.clip(np.round(x).astype(int), 0, cfg.DIMENSION-1)
        y = np.clip(np.round(y).astype(int), 0, cfg.DIMENSION-1)
        z = np.clip(z, 0, cfg.DIMENSION-1)
        
        # Set mask values
        mask[z, y, x] = 1.0
    else:
        # Regular rectangular mask
        mask[coords[0], coords[1], coords[2]] = 1.0
    
    return mask

def augment_volume(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random 3D augmentations to the volume and mask."""
    # Ensure inputs are contiguous (avoid negative strides)
    volume = np.ascontiguousarray(volume)
    mask = np.ascontiguousarray(mask)

    # Random rotation (0°, 90°, 180°, 270° around z-axis)
    if random.random() > 0.5:
        angle = random.choice([0, 90, 180, 270])
        volume = rotate(volume, angle, axes=(0, 1), reshape=False, order=1, mode='constant')
        mask = rotate(mask, angle, axes=(0, 1), reshape=False, order=0, mode='constant')
        # Force contiguous after rotation
        volume = np.ascontiguousarray(volume)
        mask = np.ascontiguousarray(mask)

    # Random flip (x, y, or z-axis)
    flip_axes = random.sample([0, 1, 2], k=random.randint(0, 3))
    for axis in flip_axes:
        volume = np.ascontiguousarray(np.flip(volume, axis=axis))
        mask = np.ascontiguousarray(np.flip(mask, axis=axis))

    # Random Gaussian noise (applied only to non-masked regions)
    if random.random() > 0.7:
        noise = np.random.normal(0, 0.05, volume.shape).astype(np.float32)
        volume = np.where(mask > 0, volume, volume + noise)
        volume = np.clip(volume, 0, 1)

    # Random intensity scaling
    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        volume = np.clip(volume * scale, 0, 1)

    return volume, mask

def create_training_pairs(volume: np.ndarray, n_augment: int = 300) -> list:
    """
    Generate training pairs with controlled distribution of mask types:
    - 40% oblique masks
    - 30% random rectangular masks
    - 30% central rectangular masks
    """
    volume = normalize_volume(volume)
    pairs = []
    
    # Calculate counts for each mask type
    n_oblique = int(0.4 * n_augment)
    n_random = int(0.3 * n_augment)
    n_central = n_augment - n_oblique - n_random  # Remaining for central
    
    # Generate oblique masks (40%)
    for _ in range(n_oblique):
        mask = create_mask(central_only=False, oblique_prob=1.0)  # Force oblique
        aug_volume, aug_mask = augment_volume(volume, mask.copy())
        conditioned = torch.from_numpy(aug_volume * aug_mask).unsqueeze(0).float()
        target = torch.from_numpy(aug_volume).unsqueeze(0).float()
        mask = torch.from_numpy(aug_mask).unsqueeze(0).float()
        pairs.append((conditioned, target, mask))
    
    # Generate random rectangular masks (30%)
    for _ in range(n_random):
        mask = create_mask(central_only=False, oblique_prob=0.0)  # Force rectangular
        aug_volume, aug_mask = augment_volume(volume, mask.copy())
        conditioned = torch.from_numpy(aug_volume * aug_mask).unsqueeze(0).float()
        target = torch.from_numpy(aug_volume).unsqueeze(0).float()
        mask = torch.from_numpy(aug_mask).unsqueeze(0).float()
        pairs.append((conditioned, target, mask))
    
    # Generate central rectangular masks (30%)
    for _ in range(n_central):
        mask = create_mask(central_only=True)  # Force central
        aug_volume, aug_mask = augment_volume(volume, mask.copy())
        conditioned = torch.from_numpy(aug_volume * aug_mask).unsqueeze(0).float()
        target = torch.from_numpy(aug_volume).unsqueeze(0).float()
        mask = torch.from_numpy(aug_mask).unsqueeze(0).float()
        pairs.append((conditioned, target, mask))
    
    # Shuffle the pairs to mix mask types
    random.shuffle(pairs)
    return pairs

def save_training_examples(conditioned: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, name: str):
    """Save sample slices for visualization (now accepts mask but doesn't use it)."""
    os.makedirs(cfg.VISUALIZATIONS_DIR, exist_ok=True)
    slice_idx = cfg.DIMENSION // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(conditioned[0, slice_idx].numpy(), cmap='gray')
    ax1.set_title("Conditioned (Masked)")
    ax2.imshow(target[0, slice_idx].numpy(), cmap='gray')
    ax2.set_title("Target")
    plt.savefig(cfg.VISUALIZATIONS_DIR / f"{name}_preview.png")
    plt.close()

def prepare_all_data():
    """Main function to prepare augmented training data."""
    print("Loading sinusoid data...")
    data = load_sinusoid_data()
    
    # Create train/val/test splits (70/15/15)
    os.makedirs(cfg.TRAIN_DATA_DIR / "train", exist_ok=True)
    os.makedirs(cfg.TRAIN_DATA_DIR / "val", exist_ok=True)
    os.makedirs(cfg.TRAIN_DATA_DIR / "test", exist_ok=True)

    for name, volume in data.items():
        print(f"\nProcessing {name} dataset...")
        pairs = create_training_pairs(volume, n_augment=80)  # Generate 10 augmented variants
        
        # Split into train/val/test (70/15/15)
        random.shuffle(pairs)
        n_train = int(0.7 * len(pairs))
        n_val = int(0.15 * len(pairs))
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:]

        # Save each split
        for split, split_pairs in zip(["train", "val", "test"], [train_pairs, val_pairs, test_pairs]):
            for i, (conditioned, target, mask) in enumerate(split_pairs):
                save_path = cfg.TRAIN_DATA_DIR / split / f"{name}_aug_{i}.pt"
                torch.save({"conditioned": conditioned, "target": target, "mask": mask}, save_path)
        
        # Save a visualization example
        save_training_examples(*train_pairs[0], name)
    
    print("\nData preparation complete! Augmented samples saved to:", cfg.TRAIN_DATA_DIR)

if __name__ == "__main__":
    prepare_all_data()