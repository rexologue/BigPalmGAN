import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

class BigGANConfig(object):
    """Configuration for BigGAN."""
    def __init__(self,
                 output_dim=512,
                 z_dim=128,
                 class_embed_dim=128,
                 channel_width=128,
                 num_classes=1000,
                 layers=None,
                 attention_layer_position=8,
                 eps=1e-4,
                 n_stats=51):
        
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        
        if layers is None:
            # Layers for 512x512 output
            # Each tuple: (up_sample, in_channels, out_channels)
            self.layers = [
                (False, 16, 16),  # 4x4
                (True, 16, 16),   # 8x8
                (False, 16, 16),  # 8x8
                (True, 16, 16),   # 16x16
                (False, 16, 16),  # 16x16
                (True, 16, 8),    # 32x32
                (False, 8, 8),    # 32x32
                (True, 8, 4),     # 64x64
                (False, 4, 4),    # 64x64
                (True, 4, 2),     # 128x128
                (False, 2, 2),    # 128x128
                (True, 2, 1),     # 256x256
                (False, 1, 1),    # 256x256
                (True, 1, 1)      # 512x512
            ]
        else:
            self.layers = layers
            
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        config = BigGANConfig()
        for key, value in json_object.items():
            setattr(config, key, value)
        return config

def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(
        nn.Conv2d(**kwargs), eps=eps)

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(
        nn.Linear(**kwargs), eps=eps)

class SelfAttn(nn.Module):
    """Self attention Layer"""
    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        
        self.in_channels = in_channels
        
        self.snconv1x1_theta = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels//8,
            kernel_size=1, bias=False, eps=eps)
        
        self.snconv1x1_phi = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels//8,
            kernel_size=1, bias=False, eps=eps)
        
        self.snconv1x1_g = snconv2d(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=1, bias=False, eps=eps)
        
        self.snconv1x1_o_conv = snconv2d(
            in_channels=in_channels//2,
            out_channels=in_channels,
            kernel_size=1, bias=False, eps=eps)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        
        attn = torch.bmm(
            theta.permute(0, 2, 1), phi)
        
        attn = self.softmax(attn)
        
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        
        attn_g = torch.bmm(
            g, attn.permute(0, 2, 1))
        
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        
        out = x + self.gamma*attn_g
        
        return out

class BigGANBatchNorm(nn.Module):
    """Batch Norm with optional conditional input."""
    def __init__(self, num_features,
                 condition_vector_dim=None,
                 n_stats=51, eps=1e-4,
                 conditional=True):
        super(BigGANBatchNorm, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.conditional = conditional
        
        self.register_buffer(
            'running_means', torch.zeros(n_stats, num_features))
        self.register_buffer(
            'running_vars', torch.ones(n_stats, num_features))
        
        self.step_size = 1.0 / (n_stats - 1)
        
        if conditional:
            assert condition_vector_dim is not None
            
            self.scale = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False, eps=eps)
            
            self.offset = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False, eps=eps)
            
        else:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))

    def forward(self, x, truncation, condition_vector=None):
        coef, start_idx = math.modf(
            truncation / self.step_size)
        
        start_idx = int(start_idx)
        
        if coef != 0.0:
            running_mean = (
                self.running_means[start_idx] * coef +
                self.running_means[start_idx + 1] * (1 - coef))
            
            running_var = (
                self.running_vars[start_idx] * coef +
                self.running_vars[start_idx + 1] * (1 - coef))
            
        else:
            running_mean = self.running_means[start_idx]
            running_var = self.running_vars[start_idx]
            
        if self.conditional:
            running_mean = running_mean.unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1)
            
            running_var = running_var.unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1)
            
            weight = 1 + self.scale(
                condition_vector).unsqueeze(-1).unsqueeze(-1)
            
            bias = self.offset(
                condition_vector).unsqueeze(-1).unsqueeze(-1)
            
            out = (x - running_mean) / torch.sqrt(
                running_var + self.eps)
            
            out = out * weight + bias
            
        else:
            out = F.batch_norm(
                x, running_mean, running_var,
                self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps)
            
        return out

class GenBlock(nn.Module):
    def __init__(self, in_size, out_size,
                 condition_vector_dim,
                 reduction_factor=4,
                 up_sample=False,
                 n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor
        
        self.bn_0 = BigGANBatchNorm(
            in_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        
        self.conv_0 = snconv2d(
            in_channels=in_size,
            out_channels=middle_size,
            kernel_size=1, eps=eps)
        
        self.bn_1 = BigGANBatchNorm(
            middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        
        self.conv_1 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3, padding=1, eps=eps)
        
        self.bn_2 = BigGANBatchNorm(
            middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        
        self.conv_2 = snconv2d(
            in_channels=middle_size,
            out_channels=middle_size,
            kernel_size=3, padding=1, eps=eps)
        
        self.bn_3 = BigGANBatchNorm(
            middle_size, condition_vector_dim,
            n_stats=n_stats, eps=eps, conditional=True)
        
        self.conv_3 = snconv2d(
            in_channels=middle_size,
            out_channels=out_size,
            kernel_size=1, eps=eps)
        
        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x
        
        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)
        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        
        if self.up_sample:
            x = F.interpolate(
                x, scale_factor=2, mode='nearest')
            
        x = self.conv_1(x)
        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)
        
        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
            
        if self.up_sample:
            x0 = F.interpolate(
                x0, scale_factor=2, mode='nearest')
            
        out = x + x0
        
        return out

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2
        
        self.gen_z = snlinear(
            in_features=condition_vector_dim,
            out_features=4 * 4 * 16 * ch, eps=config.eps)
        
        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(
                    SelfAttn(ch*layer[1], eps=config.eps))
                
            layers.append(
                GenBlock(ch*layer[1], ch*layer[2],
                         condition_vector_dim,
                         up_sample=layer[0],
                         n_stats=config.n_stats,
                         eps=config.eps))
            
        self.layers = nn.ModuleList(layers)
        
        self.bn = BigGANBatchNorm(
            ch, n_stats=config.n_stats,
            eps=config.eps, conditional=False)
        
        self.relu = nn.ReLU()
        
        self.conv_to_rgb = snconv2d(
            in_channels=ch, out_channels=3,
            kernel_size=3, padding=1, eps=config.eps)
        
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation):
        z = self.gen_z(cond_vector)
        z = z.view(-1, 16 * self.config.channel_width, 4, 4)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)
                
        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = self.tanh(z)
        
        return z

class BigGAN(nn.Module):
    """BigGAN Generator."""
    def __init__(self, config):
        super(BigGAN, self).__init__()
        
        self.config = config
        
        self.embeddings = nn.Embedding(config.num_classes, config.z_dim)

        self.generator = Generator(config)

    def forward(self, z, class_label, truncation):
        assert 0 < truncation <= 1
        
        embed = self.embeddings(class_label)
        cond_vector = torch.cat((z, embed), dim=1)
        z = self.generator(cond_vector, truncation)
        
        return z

def truncated_noise_sample(batch_size=1,
                           dim_z=128,
                           truncation=1.,
                           seed=None):
    """Create a truncated noise vector."""
    state = None if seed is None else np.random.RandomState(seed)
    
    values = truncation * \
        truncnorm.rvs(-2, 2, size=(batch_size, dim_z),
                      random_state=state)
        
    return values.astype(np.float32)
