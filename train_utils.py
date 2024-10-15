import torch
from support_utils import get_latent_input, save_sample_images
from fid.fid_score import calculate_fid

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def train_discriminator(generator, 
                        discriminator, 
                        real_images, 
                        labels, 
                        adversarial_loss, 
                        classification_loss,
                        classification_loss_weight,
                        optimizer_D,
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
    - adversarial_loss: Функция потерь для дискриминатора (обычно BCELoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - classification_loss_weight: Вес для потерь классификации.
    - optimizer_D: Оптимизатор для дискриминатора.
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - d_loss.item(): Значение потерь дискриминатора.
    """
    
    # Генерация шума и эмбеддингов классов для создания поддельных изображений
    noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, class_embeds, truncation=0.2).detach()

    # Вычисление потерь для реальных изображений с учетом меток классов
    real_binar_output, real_class_output = discriminator(real_images, labels)
    real_binar_loss = adversarial_loss(real_binar_output, torch.ones_like(real_binar_output))
    
    class_loss = classification_loss(real_class_output, labels)

    # Вычисление потерь для поддельных изображений с учетом меток классов
    fake_binar_output, _ = discriminator(fake_images, labels)
    fake_binar_loss = adversarial_loss(fake_binar_output, torch.zeros_like(fake_binar_output))

    # Среднее значение потерь для реальных и поддельных изображений
    d_loss = (real_binar_loss + fake_binar_loss) / 2 + (class_loss * classification_loss_weight)

    # Обнуление градиентов и выполнение обратного распространения ошибки
    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    return d_loss.item()

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def train_generator(generator, 
                    discriminator, 
                    real_images, 
                    labels, 
                    adversarial_loss, 
                    pixel_loss, 
                    adversarial_loss_weight,
                    pixel_loss_weight,
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
    - adversarial_loss: Функция потерь для дискриминатора (обычно BCELoss).
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - adversarial_loss_weight: Вес для потерь дискриминатора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - accumulation_steps: Количество шагов накопления градиентов перед обновлением весов.
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - g_loss.item(): Значение потерь генератора.
    """
    
    # Генерация шума и эмбеддингов классов для создания поддельных изображений
    noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
    fake_images = generator(noise, class_embeds, truncation=0.2)

    # Вычисление потерь для поддельных изображений
    fake_output, _ = discriminator(fake_images, labels)
    g_loss = (adversarial_loss(fake_output, torch.ones_like(fake_output)) * adversarial_loss_weight) + (pixel_loss(fake_images, real_images) * pixel_loss_weight)

    # Накопление градиентов
    g_loss = g_loss / accumulation_steps
    g_loss.backward()

    return g_loss.item()


################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def validate(generator, 
             discriminator, 
             inception,
             val_loader, 
             epoch, 
             img_dir,
             adversarial_loss,
             pixel_loss,
             classification_loss,
             adversarial_loss_weight,
             pixel_loss_weight,
             classification_loss_weight,
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
    - adversarial_loss: Функция потерь для дискриминатора (обычно BCELoss).
    - pixel_loss: Функция потерь для сравнения пиксельных значений (обычно L1Loss или MSELoss).
    - classification_loss: Функция потерь для классификации (обычно CrossEntropyLoss).
    - adversarial_loss_weight: Вес для потерь дискриминатора.
    - pixel_loss_weight: Вес для потерь пиксельных значений.
    - classification_loss_weight: Вес для потерь классификации.
    - num_classes: Количество классов в датасете.
    - device: Устройство (GPU/CPU), на котором происходит обучение.

    Возвращает:
    - val_g_loss: Среднее значение потерь генератора на валидационном наборе данных.
    - val_d_loss: Среднее значение потерь дискриминатора на валидационном наборе данных.
    - fid: Значение FID между реальными и сгенерированными изображениями.
    """
    
    # Переключение моделей в режим оценки
    generator.eval()
    discriminator.eval()
    
    real_fid = []
    fake_fid = []
    
    fid_amount = 30

    val_g_loss = 0
    val_d_loss = 0
    
    with torch.no_grad():
        for batch_idx, (real_images, labels) in enumerate(val_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)

            # Генерация шума и эмбеддингов классов для создания поддельных изображений
            noise, class_embeds = get_latent_input(real_images.size(0), labels, num_classes, device)
            fake_images = generator(noise, class_embeds, truncation=0.2)

            # Вычисление потерь для реальных изображений
            real_binar_output, real_class_output  = discriminator(real_images, labels)
            fake_binar_output, _ = discriminator(fake_images, labels)

            real_loss = adversarial_loss(real_binar_output, torch.ones_like(real_binar_output))
            fake_loss = adversarial_loss(fake_binar_output, torch.zeros_like(fake_binar_output))
            
            class_loss = classification_loss(real_class_output, labels)

            # Среднее значение потерь для реальных и поддельных изображений
            d_loss = (real_loss + fake_loss) / 2 + (class_loss * classification_loss_weight)
            val_d_loss += d_loss.item()

            # Вычисление потерь для генератора
            g_loss = (adversarial_loss(fake_binar_output, torch.ones_like(fake_binar_output)) * adversarial_loss_weight) + (pixel_loss(fake_images, real_images) * pixel_loss_weight)
            val_g_loss += g_loss.item()
            
            if len(real_fid) < fid_amount:
                real_fid.append(real_images)
                fake_fid.append(fake_images)

            # Сохранение сгенерированных изображений
            save_sample_images(fake_images, epoch, "validation", batch_idx, img_dir)
            
    real_fid_tensor = torch.cat(real_fid, dim=0)
    fake_fid_tensor = torch.cat(fake_fid, dim=0)
 
    fid = calculate_fid(real_fid_tensor, fake_fid_tensor, inception, device)

    # Вычисление средних значений потерь
    val_g_loss /= len(val_loader)
    val_d_loss /= len(val_loader)

    return val_g_loss, val_d_loss, fid