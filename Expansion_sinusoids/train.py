import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from config import cfg
from Unet3D_diffusion_model import DiffusionModel

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

# Disable some optimizations for stability
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SinusoidDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.data_files[idx]), weights_only=True)
        return data['conditioned'], data['target'], data['mask']

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model and optimizer
    model = DiffusionModel().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    
    # Create datasets
    train_dataset = SinusoidDataset(cfg.TRAIN_DATA_DIR, "train")
    val_dataset = SinusoidDataset(cfg.TRAIN_DATA_DIR, "val")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Setup logging and checkpointing
    log_dir = '/home/arawa/liver-3d-simulations/Expansion_sinusoids/logs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.BASE_DATA_DIR, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 100000 #currently disabled
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0
        grad_norms = []  # Track gradient norms
        
        for batch_idx, (conditioned, target, mask) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            conditioned, target, mask = conditioned.to(device), target.to(device), mask.to(device)
            
            # ===== DATA SANITY CHECK =====
            if epoch == 0 and batch_idx == 0:
                print("\nData verification:")
                print(f"Conditioned range: {conditioned.min().item():.3f} to {conditioned.max().item():.3f}")
                print(f"Target range: {target.min().item():.3f} to {target.max().item():.3f}")
                print(f"Mask unique values: {torch.unique(mask)}")
            
            # Diffusion process
            t = torch.randint(0, cfg.TIMESTEPS, (1,), device=device)
            noise = torch.randn_like(target)
            noisy_images = model.sqrt_alphas_cumprod[t] * target + model.sqrt_one_minus_alphas_cumprod[t] * noise
            
            # Forward pass
            noise_pred = model(noisy_images, t, mask)
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # ===== GRADIENT MONITORING =====
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norms.append(total_norm.item())
            
            optimizer.step()
            train_loss += loss.item()
        
        # ===== SCHEDULER STEP =====
        scheduler.step()  
        
        # Logging
        avg_train_loss = train_loss / len(train_loader)
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Gradient/Norm", avg_grad_norm, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)  # Log LR
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for conditioned, target, mask in val_loader:
                conditioned, target, mask = conditioned.to(device), target.to(device), mask.to(device)
                
                t = torch.randint(0, cfg.TIMESTEPS, (1,), device=device)
                noise = torch.randn_like(target)
                noisy_images = model.sqrt_alphas_cumprod[t] * target + model.sqrt_one_minus_alphas_cumprod[t] * noise
                
                noise_pred = model(noisy_images, t, mask)
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        
        # Checkpointing and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(cfg.BASE_DATA_DIR, "checkpoints", "best_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if epoch % 10 == 0:
            with torch.no_grad():
                conditioned, target, mask = next(iter(val_loader))
                conditioned, target, mask = conditioned.to(device), target.to(device), mask.to(device)
                
                t = torch.zeros(1, dtype=torch.long, device=device)  # Use t=0 for visualization
                noise_pred = model(conditioned, t, mask)
                
                # Visualization code
                slice_idx = conditioned.shape[2] // 2
                fig, ax = plt.subplots(1, 4, figsize=(20,5))
                
                ax[0].imshow(conditioned[0,0,slice_idx].cpu(), cmap='gray')
                ax[0].set_title("Conditioned Input")
                ax[1].imshow(target[0,0,slice_idx].cpu(), cmap='gray')
                ax[1].set_title("Ground Truth")
                ax[2].imshow(noise_pred[0,0,slice_idx].cpu(), cmap='gray')
                ax[2].set_title("Prediction")   ## fix this because prediction should look binary in itself like input and ground truth
                ax[3].imshow((torch.sigmoid(noise_pred) > 0.5).float()[0,0,slice_idx].cpu(), cmap='gray')
                ax[3].set_title("Binary Prediction")
                
                writer.add_figure("2D_Slice_Visualization", fig, epoch)
                plt.close()
    
    # Final evaluation on test set
    test_dataset = SinusoidDataset(cfg.TRAIN_DATA_DIR, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model.load_state_dict(torch.load(os.path.join(cfg.BASE_DATA_DIR, "checkpoints", "best_model.pth"))['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for conditioned, target, mask in test_loader:
            conditioned, target, mask = conditioned.to(device), target.to(device), mask.to(device)
            
            t = torch.randint(0, cfg.TIMESTEPS, (1,), device=device)
            noise = torch.randn_like(target)
            noisy_images = model.sqrt_alphas_cumprod[t] * target + model.sqrt_one_minus_alphas_cumprod[t] * noise
            
            noise_pred = model(noisy_images, t, mask)
            loss = F.mse_loss(noise_pred, noise)
            test_loss += loss.item()
            
            # For metrics
            pred_binary = (torch.sigmoid(noise_pred) > 0.5).float()
            all_preds.append(pred_binary.view(-1).cpu())
            all_targets.append(target.view(-1).cpu())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    # Calculate metrics if we have positive samples
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    all_preds_bin = (all_preds > 0.5).astype(int)
    all_targets_bin = (all_targets > 0.5).astype(int)

    if np.sum(all_targets) > 0:
        print(f"Precision: {precision_score(all_targets_bin, all_preds_bin):.4f}")
        print(f"Recall: {recall_score(all_targets_bin, all_preds_bin):.4f}")
        print(f"F1: {f1_score(all_targets_bin, all_preds_bin):.4f}")
        print(f"IoU: {jaccard_score(all_targets_bin, all_preds_bin):.4f}")
    
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    cfg.EPOCHS = 10000
    cfg.CHECKPOINT_EVERY = 10
    train()