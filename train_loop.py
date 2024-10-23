import torch
from tqdm import tqdm
from train_utils import train_discriminator, train_generator, validate
import support_utils as su

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def train(epochs,
          start_epoch,
          generator,
          discriminator,
          inception,
          optimizer_G,
          optimizer_D,
          train_loader,
          eval_loader,
          g_loss_fn,
          d_loss_fn,
          pixel_loss,
          perceptual_loss,
          g_loss_weight,
          pixel_loss_weight,
          perceptual_loss_weight,
          accumulation_steps,
          lambda_gp,
          print_every_n_batches,
          num_classes,
          img_dir,
          w_dir,
          device
          ):
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=0)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=0)

    for epoch in range(epochs):
        if epoch <= start_epoch:
            scheduler_G.step()
            scheduler_D.step()
            continue

        # Switch models to training mode
        generator.train()
        discriminator.train()

        # Initialize accumulators
        g_loss_accum = 0
        d_loss_accum = 0
        real_acc_accum = 0
        fake_acc_accum = 0

        accums = 0

        g_loss_epoch = 0
        d_loss_epoch = 0

        # Training loop
        for batch_idx, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=100)):
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Train discriminator
            d_loss, real_acc, fake_acc = train_discriminator(
                generator, discriminator, real_images, labels, d_loss_fn, optimizer_D, lambda_gp, device)

            d_loss_accum += d_loss
            real_acc_accum += real_acc
            fake_acc_accum += fake_acc

            # Train generator with gradient accumulation
            g_loss = train_generator(
                generator, discriminator, real_images, labels, inception, g_loss_fn, pixel_loss, perceptual_loss,
                g_loss_weight, pixel_loss_weight, perceptual_loss_weight, accumulation_steps, device)

            g_loss_accum += g_loss

            accums += 1

            # Update generator weights after accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

            # Save sample images
            if batch_idx % print_every_n_batches == 0 and batch_idx != 0:
                with torch.no_grad():
                    noise, labels = su.get_latent_input(real_images.size(0), labels, device)
                    fake_images = generator(noise, labels, truncation=0.4)
                    su.save_sample_images(fake_images, epoch, "train", batch_idx, img_dir)

                cur_g_loss = g_loss_accum / accums
                cur_d_loss = d_loss_accum / accums
                cur_real_acc = real_acc_accum / accums
                cur_fake_acc = fake_acc_accum / accums

                g_loss_epoch += g_loss_accum
                d_loss_epoch += d_loss_accum

                g_loss_accum = 0
                d_loss_accum = 0
                real_acc_accum = 0
                fake_acc_accum = 0
                accums = 0

                su.print_stats(epoch, batch_idx, cur_g_loss, cur_d_loss, cur_real_acc, cur_fake_acc)

            su.clear_cache()

        # Validation at the end of the epoch
        val_g_loss, val_d_loss, fid, val_real_acc, val_fake_acc = validate(
            generator, discriminator, inception, eval_loader, epoch, img_dir, g_loss_fn, d_loss_fn,
            pixel_loss, perceptual_loss, g_loss_weight, pixel_loss_weight, perceptual_loss_weight, num_classes, device)

        su.print_stats(epoch, 'end', g_loss_epoch / len(train_loader), d_loss_epoch / len(train_loader), val_real_acc, val_fake_acc, val_g_loss, val_d_loss, fid)

        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Save checkpoints
        su.save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, w_dir)
