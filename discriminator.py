import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DiscriminatorBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        
        self.skip_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            
        skip = self.skip_conv(x)
        
        if self.downsample:
            skip = F.avg_pool2d(skip, 2)
            
        return out + skip

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
class SelfAttn(nn.Module):
    """ Self-attention Layer"""
    def __init__(self, in_channels):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        
        self.snconv1x1_theta = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False))
        self.snconv1x1_phi = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False))
        self.snconv1x1_g = spectral_norm(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False))
        self.snconv1x1_o_conv = spectral_norm(nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False))
        
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, ch, h, w = x.size()
        theta = self.snconv1x1_theta(x).view(batch_size, -1, h * w)
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi).view(batch_size, -1, h * w // 4)
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        g = self.snconv1x1_g(x)
        g = self.maxpool(g).view(batch_size, -1, h * w // 4)
        attn_g = torch.bmm(g, attn.permute(0, 2, 1)).view(batch_size, ch // 2, h, w)
        out = x + self.gamma * self.snconv1x1_o_conv(attn_g)
        return out

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = 64  # Reduced embedding dimension
        self.embed = nn.Embedding(num_classes, self.embed_dim)
        
        self.feature_map_dim = 16

        self.initial_conv = spectral_norm(nn.Conv2d(3, self.feature_map_dim, kernel_size=3, padding=1))
        
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(self.feature_map_dim, self.feature_map_dim * 2, downsample=True),       # 256x256 -> 128x128
            DiscriminatorBlock(self.feature_map_dim * 2, self.feature_map_dim * 4, downsample=True),   # 128x128 -> 64x64
            DiscriminatorBlock(self.feature_map_dim * 4, self.feature_map_dim * 4, downsample=True),   # 64x64 -> 32x32
            DiscriminatorBlock(self.feature_map_dim * 4, self.feature_map_dim * 4, downsample=True),   # 32x32 -> 16x16
            DiscriminatorBlock(self.feature_map_dim * 4, self.feature_map_dim * 8, downsample=True),   # 16x16 -> 8x8
            DiscriminatorBlock(self.feature_map_dim * 8, self.feature_map_dim * 16, downsample=True),  # 8x8 -> 4x4
            DiscriminatorBlock(self.feature_map_dim * 16, self.embed_dim, downsample=False),           # 4x4 -> 4x4
        ])
        
        self.self_attn = SelfAttn(self.feature_map_dim * 4)  # Insert attention layer after the second block
        self.activation = nn.LeakyReLU(0.1)
        
        self.fc = spectral_norm(nn.Linear(self.embed_dim, 1))

    def forward(self, x, labels):
        h = self.initial_conv(x)
        
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if idx == 1:  # Apply self-attention after the second block
                h = self.self_attn(h)
                
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])  # Global sum pooling, resulting in [batch_size, embed_dim]
        
        out_adv = self.fc(h).squeeze(1)  # Output scalar value per batch
        
        # Projection discriminator
        embed = self.embed(labels)  # Shape: [batch_size, embed_dim]
        proj = torch.sum(h * embed, dim=1)
        out_adv += proj
        
        return out_adv
