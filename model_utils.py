import torch
import torch.nn as nn
from biggan import BigGAN, BigGANConfig, GenBlock, SelfAttn
from discriminator import Discriminator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def instance_generator(num_classes):
    """Creates an instance of the BigGAN generator."""
    confd = {"attention_layer_position": 8,
            "channel_width": 128,
            "class_embed_dim": 128,
            "eps": 0.0001,
            "layers":  [[False, 16, 16],
                        [True, 16, 16],
                        [False, 16, 16],
                        [True, 16, 8],
                        [False, 8, 8],
                        [True, 8, 8],
                        [False, 8, 8],
                        [True, 8, 4],
                        [False, 4, 4],
                        [True, 4, 2],
                        [False, 2, 2],
                        [True, 2, 1],
                        [False, 1, 1],
                        [True, 1, 1]],
            "n_stats": 51,
            "num_classes": num_classes,
            "output_dim": 512,
            "z_dim": 128
            }
    
    conf = BigGANConfig.from_dict(confd)
    
    return BigGAN(conf)

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def load_generator(num_classes, ckp):
    """Loads the generator's state from a checkpoint."""
    generator = instance_generator(num_classes)
    
    if isinstance(ckp, str):
        generator.load_state_dict(torch.load(ckp))
    else:
        generator.load_state_dict(ckp)
        
    return generator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def setup_generator(num_classes, unfreeze_last_n, ckp=None):
    """Sets up the generator, loading from checkpoint if provided."""
    
    if ckp is not None:
        generator = load_generator(num_classes, ckp)
    else:
        generator = instance_generator(num_classes)
        
    freeze_generator(generator, unfreeze_last_n)
    
    return generator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def freeze_generator(generator, unfreeze_last_n=-1, unfreeze_embeddings=True):
    """
    Freezes layers of the generator except for the last `unfreeze_last_n` GenBlocks.
    If `unfreeze_last_n` is -1, all layers are unfrozen.

    Parameters:
    - generator: The BigGAN generator model.
    - unfreeze_last_n: Number of last GenBlocks to unfreeze. If -1, unfreezes all layers.
    - unfreeze_embeddings: Whether to unfreeze the embeddings layer.
    """
    # Unfreeze embeddings if specified
    if unfreeze_embeddings:
        for param in generator.embeddings.parameters():
            param.requires_grad = True
    else:
        for param in generator.embeddings.parameters():
            param.requires_grad = False

    # Get all GenBlocks
    gen_blocks = [layer for layer in generator.generator.layers if isinstance(layer, GenBlock)]

    if unfreeze_last_n == -1:
        # Unfreeze all parameters
        for param in generator.parameters():
            param.requires_grad = True
    else:
        # Freeze all parameters
        for param in generator.parameters():
            param.requires_grad = False

        # Unfreeze the last `unfreeze_last_n` GenBlocks
        for block in gen_blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze Self-Attention layer
        for layer in generator.generator.layers:
            if isinstance(layer, SelfAttn):
                for param in layer.parameters():
                    param.requires_grad = True

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def setup_discriminator(num_classes, ckp=None):
    """Creates an instance of the discriminator and initializes weights."""
    
    discriminator = Discriminator(num_classes)
    discriminator.apply(initialize_weights)
    
    if ckp is not None:
        discriminator.load_state_dict(ckp)
        
    return discriminator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def initialize_weights(m):
    """Initializes weights of the model."""
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)
