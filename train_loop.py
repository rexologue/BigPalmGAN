import torch
from tqdm import tqdm

from train_utils import train_discriminator, train_generator, validate
import support_utils as su

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
          classification_loss,
          perceptual_loss,
          g_loss_weight,
          pixel_loss_weight,
          classification_loss_weight,
          perceptual_loss_weight,
          accumulation_steps,
          lambda_gp,
          print_every_n_batches,
          num_classes,
          img_dir,
          w_dir,
          device
          ):
    """
    Обучает генератор и дискриминатор на заданном количестве эпох.

    Параметры:
    - epochs: Количество эпох для обучения.
    - start_epoch: Начальная эпоха (используется при загрузке чекпоинтов).
    - generator: Генератор, который обучается создавать поддельные изображения.
    - discriminator: Дискриминатор, который обучается различать реальные и поддельные изображения.
    - inception: Модель Inception для вычисления FID.
    - optimizer_G: Оптимизатор для генератора.
    - optimizer_D: Оптимизатор для дискриминатора.
    - train_loader: Загрузчик данных для обучения.
    - eval_loader: Загрузчик данных для валидации.
    - g_loss_fn: Функция потерь для генератора.
    - d_loss_fn: Функция потерь для дискриминатора.
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - perceptual_loss: Функция перцептивных потерь.
    - g_loss_weight: Вес для потерь генератора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - classification_loss_weight: Вес для потерь классификации.
    - perceptual_loss_weight: Вес для потерь перцептивных значений.
    - accumulation_steps: Количество шагов накопления градиентов перед обновлением весов.
    - print_every_n_batches: Частота вывода статистики во время обучения.
    - num_classes: Количество классов в датасете.
    - img_dir: Директория для сохранения сгенерированных изображений.
    - w_dir: Директория для сохранения чекпоинтов.
    - device: Устройство (GPU/CPU), на котором происходит обучение.
    """
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=0)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=0)
    
    for epoch in range(epochs):
        if epoch <= start_epoch:
            scheduler_G.step()
            scheduler_D.step()
            continue
        
        # Переключение моделей в режим обучения
        generator.train()
        discriminator.train()

        # Инициализация аккумуляторов потерь и точности
        g_loss_accum = 0
        d_loss_accum = 0
        real_acc_accum = 0
        fake_acc_accum = 0
        
        accums = 0
        
        g_loss_epoch = 0
        d_loss_epoch = 0

        # Обучение на каждом батче данных
        for batch_idx, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=100)):
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Обучение дискриминатора
            d_loss, real_acc, fake_acc = train_discriminator(generator, discriminator, real_images, labels, d_loss_fn, classification_loss, 
                                                             classification_loss_weight, optimizer_D, lambda_gp, num_classes, device)
            
            d_loss_accum += d_loss
            real_acc_accum += real_acc
            fake_acc_accum += fake_acc

            # Обучение генератора с накоплением градиентов
            g_loss = train_generator(generator, discriminator, real_images, labels, inception, g_loss_fn, pixel_loss, perceptual_loss,
                                      g_loss_weight, pixel_loss_weight, perceptual_loss_weight, accumulation_steps, num_classes, device)

            g_loss_accum += g_loss
            
            accums += 1

            # Обновление весов генератора после накопления градиентов
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

            # Сохранение сгенерированных изображений каждые print_every_n_batches батчей
            if batch_idx % print_every_n_batches == 0 and batch_idx != 0:
                with torch.no_grad():
                    noise, class_embeds = su.get_latent_input(real_images.size(0), labels, num_classes, device)
                    fake_images = generator(noise, class_embeds, truncation=0.2)
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

                # Вывод статистики каждые print_every_n_batches батчей
                su.print_stats(epoch, batch_idx, cur_g_loss, cur_d_loss, cur_real_acc, cur_fake_acc)

            # Очистка кеша
            su.clear_cache()

        # Валидация в конце каждой эпохи
        val_g_loss, val_d_loss, fid, val_real_acc, val_fake_acc = validate(generator, discriminator, inception, eval_loader, epoch, img_dir, g_loss_fn, d_loss_fn, 
                                                                           pixel_loss, classification_loss, perceptual_loss, g_loss_weight, pixel_loss_weight, 
                                                                           classification_loss_weight, perceptual_loss_weight, num_classes, device)
        
        su.print_stats(epoch, 'end', g_loss_epoch / len(train_loader), d_loss_epoch / len(train_loader), val_real_acc, val_fake_acc, val_g_loss, val_d_loss, fid)

        # Обновление планировщиков скорости обучения
        scheduler_G.step()
        scheduler_D.step()

        # Сохранение чекпоинтов
        su.save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, w_dir)