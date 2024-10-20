import torch
from support_utils import get_latent_input, save_sample_images
from fid.fid_score import calculate_fid

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def compute_gradient_penalty(discriminator, real_images, fake_images, labels, device):
    # Random weight for interpolation
    alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
    
    # Interpolation between real and fake images
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    
    # Discriminator's output for interpolated images
    d_interpolates, _ = discriminator(interpolates, labels)
    
    # Gradients of d_interpolates with respect to interpolated images
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
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
                        device
                        ):
    """
    Обучает дискриминатор на одном батче реальных и сгенерированных изображений.

    Параметры:
    - generator: Генератор, используемый для создания поддельных изображений.
    - discriminator: Дискриминатор, который обучается различать реальные и поддельные изображения.
    - real_images: Батч реальных изображений.
    - labels: Метки классов для реальных изображений.
    - d_loss_fn: Функция потерь для дискриминатора (обычно HingeLoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - classification_loss_weight: Вес для потерь классификации.
    - optimizer_D: Оптимизатор для дискриминатора.
    - lambda_gp: Значение коэффициента gradient penalty
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - d_loss.item(): Значение потерь дискриминатора.
    - real_acc.item(): Точность для реальных изображений.
    - fake_acc.item(): Точность для поддельных изображений.
    """
    
    # Генерация фейковых изображений
    noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, class_embeds, truncation=0.2)

    # Предсказания дискриминатора для реальных и фейковых изображений
    real_outputs, real_class_outputs = discriminator(real_images, labels)  # Реальные изображения и предсказания классов
    fake_outputs, _ = discriminator(fake_images.detach(), labels)  # Фейковые изображения (без обновления генератора)

    # Потери дискриминатора по HingeLoss
    d_hinge_loss = d_loss_fn(real_outputs, fake_outputs)  # Передаем реальные и фейковые выходы одновременно в HingeLoss

    # Классификационные потери для реальных изображений
    d_class_loss = classification_loss(real_class_outputs, labels) * classification_loss_weight
    
    # Вычисляем Gradient Penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, labels, device)
    
    # Итоговые потери дискриминатора с Gradient Penalty
    d_loss = d_hinge_loss + d_class_loss + lambda_gp * gradient_penalty

    # Обновление весов дискриминатора
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    # Подсчет точности для реальных и фейковых изображений
    real_acc = (real_outputs > 0).float().mean()
    fake_acc = (fake_outputs < 0).float().mean()

    return d_loss.item(), real_acc.item(), fake_acc.item()

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
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
                    accumulation_steps,
                    num_classes,
                    device
                    ):
    """
    Обучает генератор на одном батче реальных и сгенерированных изображений.

    Параметры:
    - generator: Генератор, который обучается создавать поддельные изображения.
    - discriminator: Дискриминатор, используемый для оценки поддельных изображений.
    - real_images: Батч реальных изображений.
    - labels: Метки классов для реальных изображений.
    - g_loss_fn: Функция потерь для генератора (обычно HingeLoss).
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - perceptual_loss: Функция перцептивных потерь.
    - g_loss_weight: Вес для потерь генератора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - perceptual_loss_weight: Вес для потерь перцептивных значений.
    - accumulation_steps: Количество шагов накопления градиентов перед обновлением весов.
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - g_loss.item(): Значение потерь генератора.
    """
    
    # Генерация фейковых изображений
    noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, class_embeds, truncation=0.2)

    # Предсказания дискриминатора для фейковых изображений
    fake_outputs, _ = discriminator(fake_images, labels)

    # Расчет функций потерь для генератора
    g_hinge_loss = g_loss_fn(fake_outputs) * g_loss_weight
    g_pixel_loss = pixel_loss(fake_images, real_images) * pixel_loss_weight
    g_percep_loss = perceptual_loss(inception, fake_images, real_images) * perceptual_loss_weight

    g_loss = g_hinge_loss + g_pixel_loss + g_percep_loss

    # Gradient accumulation
    g_loss = g_loss / accumulation_steps
    g_loss.backward()

    return g_hinge_loss.item() + g_pixel_loss.item() + g_percep_loss.item()

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def validate(generator, 
             discriminator, 
             inception,
             val_loader, 
             epoch, 
             img_dir,
             g_loss_fn,
             d_loss_fn,
             pixel_loss,
             classification_loss,
             perceptual_loss,
             g_loss_weight,
             pixel_loss_weight,
             classification_loss_weight,
             perceptual_loss_weight,
             num_classes,
             device
             ):
    """
    Валидация модели на валидационном наборе данных с сохранением сгенерированных изображений.

    Параметры:
    - generator: Генератор, используемый для создания поддельных изображений.
    - discriminator: Дискриминатор, используемый для оценки поддельных изображений.
    - inception: Модель Inception для вычисления FID.
    - val_loader: Загрузчик данных для валидации.
    - epoch: Текущая эпоха обучения.
    - img_dir: Директория для сохранения сгенерированных изображений.
    - g_loss_fn: Функция потерь для генератора (обычно HingeLoss).
    - d_loss_fn: Функция потерь для дискриминатора (обычно HingeLoss).
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - perceptual_loss: Функция перцептивных потерь.
    - g_loss_weight: Вес для потерь генератора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - classification_loss_weight: Вес для потерь классификации.
    - perceptual_loss_weight: Вес для потерь перцептивных значений.
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - val_g_loss: Среднее значение потерь генератора на валидационном наборе данных.
    - val_d_loss: Среднее значение потерь дискриминатора на валидационном наборе данных.
    - fid: Значение FID между реальными и сгенерированными изображениями.
    - real_acc: Средняя точность для реальных изображений.
    - fake_acc: Средняя точность для поддельных изображений.
    """
    
    # Переключение моделей в режим оценки
    generator.eval()
    discriminator.eval()
    
    real_fid = []
    fake_fid = []
    fid_amount = 30

    val_g_loss = 0
    val_d_loss = 0
    real_acc = 0
    fake_acc = 0
    
    with torch.no_grad():
        for real_images, labels in val_loader:
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Генерация поддельных изображений
            noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
            fake_images = generator(noise, class_embeds, truncation=0.2)

            # Получение предсказаний дискриминатора для реальных и поддельных изображений
            real_outputs, real_class_outputs = discriminator(real_images, labels)
            fake_outputs, _ = discriminator(fake_images.detach(), labels)

            # Hinge loss для реальных и поддельных изображений
            d_hinge_loss = d_loss_fn(real_outputs, fake_outputs)

            # Потери классификации для реальных изображений
            d_class_loss = classification_loss(real_class_outputs, labels) * classification_loss_weight

            # Общие потери дискриминатора
            d_loss = d_hinge_loss + d_class_loss
            val_d_loss += d_loss.item()

            # Расчет функций потерь для генератора
            g_hinge_loss = g_loss_fn(fake_outputs) * g_loss_weight
            g_pixel_loss = pixel_loss(fake_images, real_images) * pixel_loss_weight
            g_percep_loss = perceptual_loss(inception, fake_images, real_images) * perceptual_loss_weight
            
            g_loss = g_hinge_loss + g_pixel_loss + g_percep_loss
            val_g_loss += g_loss.item()
            
            if len(real_fid) < fid_amount:
                real_fid.append(real_images)
                fake_fid.append(fake_images)
                
            # Подсчет точности для реальных и фейковых изображений
            real_acc += (real_outputs > 0).float().mean()
            fake_acc += (fake_outputs < 0).float().mean()

        # Сохранение сгенерированных изображений
        save_sample_images(fake_images, epoch, "validation", "end", img_dir)

    # Рассчитываем FID
    real_fid_tensor = torch.cat(real_fid, dim=0)
    fake_fid_tensor = torch.cat(fake_fid, dim=0)
    fid = calculate_fid(real_fid_tensor, fake_fid_tensor, inception, device)

    # Средние значения потерь и точности
    val_g_loss /= len(val_loader)
    val_d_loss /= len(val_loader)
    real_acc /= len(val_loader)
    fake_acc /= len(val_loader)

    return val_g_loss, val_d_loss, fid, real_acc, fake_acc