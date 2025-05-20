import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, time):
        emb = time[:, None] * self.emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        diff = [skip.size()[i+2] - x.size()[i+2] for i in range(3)]
        x = F.pad(x, [d // 2 for d in diff] + [d - d//2 for d in diff])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_dim=64):  # Changed in_channels to 2 (image + mask)
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        
        # Bottleneck
        self.bottleneck = DoubleConv(64, 128)
        
        # Decoder
        self.up1 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up3 = UpBlock(32, 16)
        
        self.outc = nn.Conv3d(16, out_channels, 1)
        
    def forward(self, x, time, mask):
        # Combine input with mask along channel dimension
        x = torch.cat([x, mask], dim=1)
        t = self.time_mlp(time)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        # Bottleneck
        x = self.bottleneck(x3)
        
        # Decoder
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return self.outc(x)

class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet3D()
        self.timesteps = cfg.TIMESTEPS
        
        betas = torch.linspace(1e-4, 0.02, self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def forward(self, x, time, mask):
        return self.unet(x, time, mask)