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
          adversarial_loss,
          pixel_loss,
          classification_loss,
          adversarial_loss_weight,
          pixel_loss_weight,
          classification_loss_weight,
          accumulation_steps,
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
    - adversarial_loss: Функция потерь для дискриминатора (обычно BCELoss).
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - adversarial_loss_weight: Вес для потерь дискриминатора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - classification_loss_weight: Вес для потерь классификации.
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

        # Инициализация аккумуляторов потерь
        g_loss_accum = 0
        d_loss_accum = 0

        # Обучение на каждом батче данных
        for batch_idx, (real_images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=100)):
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Обучение дискриминатора
            d_loss = train_discriminator(generator, discriminator, real_images, labels, adversarial_loss, classification_loss, classification_loss_weight, optimizer_D, num_classes, device)
            d_loss_accum += d_loss

            # Обучение генератора с накоплением градиентов
            g_loss = 0
            for _ in range(accumulation_steps):
                g_loss += train_generator(generator, discriminator, real_images, labels, adversarial_loss, pixel_loss, 
                                          adversarial_loss_weight, pixel_loss_weight, accumulation_steps, num_classes, device)

            g_loss_accum += g_loss

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

                # Вывод статистики каждые print_every_n_batches батчей
                su.print_stats(epoch, batch_idx, g_loss_accum / (batch_idx + 1), d_loss_accum / (batch_idx + 1))

            # Очистка кеша
            su.clear_cache()

        # Валидация в конце каждой эпохи
        val_g_loss, val_d_loss, fid = validate(generator, discriminator, inception, eval_loader, epoch, img_dir, adversarial_loss, pixel_loss, classification_loss,
                                               adversarial_loss_weight, pixel_loss_weight, classification_loss_weight, num_classes, device)
        
        su.print_stats(epoch, 'end', g_loss_accum / len(train_loader), d_loss_accum / len(train_loader), val_g_loss, val_d_loss, fid)

        # Обновление планировщиков скорости обучения
        scheduler_G.step()
        scheduler_D.step()

        # Сохранение чекпоинтов
        su.save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, w_dir)