import gc
import os
import torch
import numpy as np
from biggan import truncated_noise_sample
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def get_latent_input(batch_size, labels, device='cpu'):
    noise = truncated_noise_sample(truncation=0.2, batch_size=batch_size).to(device)
    labels = labels.to(device).long()
    return noise, labels    

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def save_sample_images(images, epoch, phase, batch_idx, save_dir):
    grid = make_grid(images[:8], nrow=4, normalize=True)
    
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Epoch {epoch}, Batch {batch_idx}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f'{phase}_epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def save_sample_images_by_class(class_fake_images, epoch, save_dir, num_classes):
    """
    Saves a grid of images, one per class.

    Parameters:
    - class_fake_images: Dictionary with one fake image per class.
    - epoch: Current epoch number.
    - save_dir: Directory to save images.
    - num_classes: Total number of classes.
    """
    
    images = []
    labels = []

    # Collect images and labels in order of class IDs
    for class_id in range(num_classes):
        if class_id in class_fake_images:
            images.append(class_fake_images[class_id])
            labels.append(f'Class {class_id}')
            
        else:
            # If no image was generated for a class, create a placeholder
            placeholder = torch.zeros_like(next(iter(class_fake_images.values())))
            images.append(placeholder)
            labels.append(f'Class {class_id} (N/A)')

    if images:
        # Create a grid of images
        grid = make_grid(images, nrow=num_classes, normalize=True, scale_each=True)
        
        plt.figure(figsize=(num_classes * 2, 2))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title(f"Epoch {epoch} - Validation Samples")
        plt.axis('off')
        
        # Annotate each image with its class label
        for i, label in enumerate(labels):
            plt.text(i * grid.shape[2] / num_classes + 5, grid.shape[1] - 10, label, color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
            
        plt.savefig(os.path.join(save_dir, f'validation_epoch_{epoch}.png'))
        plt.close()


################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def print_stats(epoch, batch_idx, g_loss, d_loss, real_acc, fake_acc, val_g_loss=None, val_d_loss=None, fid=None):
    print(f"\nEpoch [{epoch}], Batch [{batch_idx}]")
    print(f"Training - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Real Accuracy: {real_acc:.4f}, Fake Accuracy: {fake_acc:.4f}")
    
    if val_g_loss is not None and val_d_loss is not None and fid is not None:
        print(f"Validation - G Loss: {val_g_loss:.4f}, D Loss: {val_d_loss:.4f}, FID: {fid:.4f}, Real Accuracy: {real_acc:.4f}, Fake Accuracy: {fake_acc:.4f}")
        
    print('\n')


################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, path):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G is not None else None,
        'optimizer_D_state_dict': optimizer_D.state_dict() if optimizer_D is not None else None
    }, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))


################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def clear_cache():
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  gc.collect()

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def create_unique_directory(base_name):
    counter = 1
    dir_name = base_name

    while os.path.exists(dir_name):
        dir_name = f"{base_name}_{counter}"
        counter += 1

    os.makedirs(dir_name)
    return dir_name

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False