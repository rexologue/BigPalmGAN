import torch
from support_utils import get_latent_input, save_sample_images_by_class
from fid.fid_score import calculate_fid

def compute_gradient_penalty(discriminator, real_images, fake_images, labels, device):
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_discriminator(generator,
                        discriminator,
                        real_images,
                        labels,
                        d_loss_fn,
                        classification_loss,
                        classification_loss_weight,
                        optimizer_D,
                        lambda_gp,
                        num_classes,
                        device):
    # Generate fake images
    noise, labels = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, labels, truncation=0.4)

    # Get discriminator outputs
    real_outputs = discriminator(real_images, labels)
    fake_outputs = discriminator(fake_images.detach(), labels)

    # Discriminator losses
    d_adv_loss = d_loss_fn(real_outputs, fake_outputs)

    # Classification loss (for real images)
    # Assume you have a classification head in the discriminator
    # If not, you can skip this loss or modify the discriminator accordingly
    # d_class_outputs = discriminator.classify(real_images)
    # d_class_loss = classification_loss(d_class_outputs, labels) * classification_loss_weight
    # Total discriminator loss
    d_loss = d_adv_loss  # + d_class_loss (if applicable)

    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images.detach(), labels, device)
    d_loss += lambda_gp * gradient_penalty

    # Backpropagation and optimization
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Compute accuracies
    real_acc = (real_outputs > 0).float().mean()
    fake_acc = (fake_outputs < 0).float().mean()

    return d_loss.item(), real_acc.item(), fake_acc.item()

def train_generator(generator,
                    discriminator,
                    real_images,
                    labels,
                    inception,
                    g_loss_fn,
                    pixel_loss,
                    perceptual_loss,
                    g_loss_weight,
                    pixel_loss_weight,
                    perceptual_loss_weight,
                    classification_loss,
                    classification_loss_weight,
                    accumulation_steps,
                    num_classes,
                    device):
    # Generate fake images
    noise, labels = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, labels, truncation=0.4)

    # Get discriminator outputs
    fake_outputs = discriminator(fake_images, labels)

    # Generator adversarial loss
    g_adv_loss = g_loss_fn(fake_outputs) * g_loss_weight

    # Pixel loss
    g_pixel_loss = pixel_loss(fake_images, real_images) * pixel_loss_weight

    # Perceptual loss
    g_percep_loss = perceptual_loss(inception, fake_images, real_images) * perceptual_loss_weight

    # Classification loss (using discriminator's projection)
    # If the discriminator provides class outputs, you can use them
    # Otherwise, you might need to adjust this part
    # fake_class_outputs = discriminator.classify(fake_images)
    # g_class_loss = classification_loss(fake_class_outputs, labels) * classification_loss_weight

    # Total generator loss
    g_loss = g_adv_loss + g_pixel_loss + g_percep_loss  # + g_class_loss (if applicable)

    # Gradient accumulation
    g_loss = g_loss / accumulation_steps
    g_loss.backward()

    return g_adv_loss.item() + g_pixel_loss.item() + g_percep_loss.item()  # + g_class_loss.item() (if applicable)

def validate(generator,
             discriminator,
             inception,
             val_loader,
             epoch,
             img_dir,
             g_loss_fn,
             d_loss_fn,
             pixel_loss,
             perceptual_loss,
             g_loss_weight,
             pixel_loss_weight,
             perceptual_loss_weight,
             num_classes,
             device):
    generator.eval()
    discriminator.eval()

    real_fid = []
    fake_fid = []

    val_g_loss = 0
    val_d_loss = 0
    real_acc = 0
    fake_acc = 0

    # Initialize a dictionary to store one fake image per class
    class_fake_images = {}

    with torch.no_grad():
        for real_images, labels in val_loader:
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Generate fake images
            batch_size = real_images.size(0)
            noise, labels = get_latent_input(batch_size, labels, num_classes, device)
            fake_images = generator(noise, labels, truncation=0.4)

            # Get discriminator outputs
            real_outputs = discriminator(real_images, labels)
            fake_outputs = discriminator(fake_images, labels)

            # Discriminator losses
            d_adv_loss = d_loss_fn(real_outputs, fake_outputs)
            d_loss = d_adv_loss
            val_d_loss += d_loss.item()

            # Generator losses
            g_adv_loss = g_loss_fn(fake_outputs) * g_loss_weight
            g_pixel_loss = pixel_loss(fake_images, real_images) * pixel_loss_weight
            g_percep_loss = perceptual_loss(inception, fake_images, real_images) * perceptual_loss_weight
            g_loss = g_adv_loss + g_pixel_loss + g_percep_loss
            val_g_loss += g_loss.item()

            # Collect images for FID calculation
            real_fid.append(real_images.cpu())
            fake_fid.append(fake_images.cpu())

            # Compute accuracies
            real_acc += (real_outputs > 0).float().mean().item()
            fake_acc += (fake_outputs < 0).float().mean().item()

            # Collect one image per class
            for img, label in zip(fake_images, labels):
                label = label.item()
                if label not in class_fake_images:
                    class_fake_images[label] = img.cpu()
                # Break the loop if we have collected all classes
                if len(class_fake_images) == num_classes:
                    break
            # Break outer loop if we have collected all classes
            if len(class_fake_images) == num_classes:
                break

        # Save sample images per class
        save_sample_images_by_class(class_fake_images, epoch, img_dir, num_classes)

    # Calculate FID
    real_fid_tensor = torch.cat(real_fid, dim=0)
    fake_fid_tensor = torch.cat(fake_fid, dim=0)
    fid = calculate_fid(real_fid_tensor, fake_fid_tensor, inception, device)

    # Average losses and accuracies
    num_batches = len(val_loader)
    val_g_loss /= num_batches
    val_d_loss /= num_batches
    real_acc /= num_batches
    fake_acc /= num_batches

    return val_g_loss, val_d_loss, fid, real_acc, fake_acc

