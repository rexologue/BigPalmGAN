import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

class BigGANConfig(object):
    def __init__(self,
                 output_dim=128,
                 z_dim=128,
                 class_embed_dim=128,
                 channel_width=128,
                 num_classes=1000,
                 layers=[(False, 16, 16),
                         (True, 16, 16),
                         (False, 16, 16),
                         (True, 16, 8),
                         (False, 8, 8),
                         (True, 8, 4),
                         (False, 4, 4),
                         (True, 4, 2),
                         (False, 2, 2),
                         (True, 2, 1)],
                 attention_layer_position=8,
                 eps=1e-4,
                 n_stats=51):
        """Constructs BigGANConfig. """
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
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
                 n_stats=51,
                 eps=1e-4,
                 momentum=0.1,
                 conditional=True):
        super(BigGANBatchNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.conditional = conditional

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        if self.conditional:
            assert condition_vector_dim is not None

            self.scale = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False, eps=eps)

            self.offset = snlinear(
                in_features=condition_vector_dim,
                out_features=num_features,
                bias=False, eps=eps)

            # Initialize scale and offset parameters
            nn.init.zeros_(self.scale.weight)
            nn.init.zeros_(self.offset.weight)
        else:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, truncation, condition_vector=None):
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean([0, 2, 3])  # Mean over batch, height, width
            batch_var = x.var([0, 2, 3], unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
                self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            var = self.running_var

        # Reshape mean and var for broadcasting
        mean = mean.view(1, self.num_features, 1, 1)
        var = var.view(1, self.num_features, 1, 1)

        if self.conditional:
            assert condition_vector is not None, "Condition vector is required for conditional batch norm"

            # Compute gamma and beta from condition_vector
            gamma = self.scale(condition_vector).view(-1, self.num_features, 1, 1) + 1  # +1 for residual scaling
            beta = self.offset(condition_vector).view(-1, self.num_features, 1, 1)

            # Normalize x
            x = (x - mean) / torch.sqrt(var + self.eps)
            out = gamma * x + beta
        else:
            # Standard batch normalization with learnable parameters
            weight = self.weight.view(1, self.num_features, 1, 1)
            bias = self.bias.view(1, self.num_features, 1, 1)
            x = (x - mean) / torch.sqrt(var + self.eps)
            out = weight * x + bias

        return out


class GenBlock(nn.Module):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)

        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
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
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        out = x + x0
        return out

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        ch = config.channel_width
        condition_vector_dim = config.z_dim * 2

        self.gen_z = snlinear(in_features=condition_vector_dim,
                              out_features=4 * 4 * 16 * ch, eps=config.eps)

        layers = []
        for i, layer in enumerate(config.layers):
            if i == config.attention_layer_position:
                layers.append(SelfAttn(ch*layer[1], eps=config.eps))
            layers.append(GenBlock(ch*layer[1],
                                   ch*layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=config.n_stats,
                                   eps=config.eps))
        self.layers = nn.ModuleList(layers)

        self.bn = BigGANBatchNorm(ch, n_stats=config.n_stats, eps=config.eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=config.eps)
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation):
        z = self.gen_z(cond_vector)

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.config.channel_width)
        z = z.permute(0, 3, 1, 2).contiguous()

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)

        z = self.bn(z, truncation)
        z = self.relu(z)
        z = self.conv_to_rgb(z)
        z = z[:, :3, ...]
        z = self.tanh(z)
        return z


class BigGAN(nn.Module):
    """BigGAN Generator."""
    def __init__(self, config):
        super(BigGAN, self).__init__()
        
        self.config = config
        
        self.embeddings = nn.Embedding(config.num_classes, config.z_dim)

        self.generator = Generator(config)

    def forward(self, z, class_label): 
        embed = self.embeddings(class_label)
        cond_vector = torch.cat((z, embed), dim=1)
        z = self.generator(cond_vector)
        
        return z

def truncated_noise_sample(batch_size, dim_z, truncation, device):
    noise = torch.randn(batch_size, dim_z, device=device)
    if truncation < 1.0:
        noise = torch.clamp(noise, -truncation, truncation)
    return noise

