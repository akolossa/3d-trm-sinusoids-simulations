import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.Conv3d(64, 64, kernel_size=2, stride=2)  # Downsampling
        self.enc2 = conv_block(64, 128)

        # Removed the third pooling layer to reduce the downsampling
        self.enc3 = conv_block(128, 256)
        self.pool2 = nn.Conv3d(256, 256, kernel_size=2, stride=2)

        self.enc4 = conv_block(256, 512)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        self.dec4 = conv_block(1024 + 512, 512)
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)

        self.conv_final = nn.Conv3d(64, out_channels, kernel_size=1)

    def upsample(self, x, target_size):
        return F.interpolate(x, size=target_size, mode='trilinear')

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc1p = self.pool1(enc1)  # Pooling after enc1

        enc2 = self.enc2(enc1p)

        enc3 = self.enc3(enc2)
        enc3p = self.pool2(enc3)  # Pooling after enc3

        enc4 = self.enc4(enc3p)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoding path
        dec4 = self.upsample(bottleneck, enc4.size()[2:])
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upsample(dec4, enc3.size()[2:])
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upsample(dec3, enc2.size()[2:])
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upsample(dec2, enc1.size()[2:])
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Final convolution
        output = self.conv_final(dec1)

        return output
