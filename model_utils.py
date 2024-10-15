import torch
import torch.nn as nn
from biggan import BigGAN, BigGANConfig, GenBlock
from discriminator import Discriminator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def instance_generator(num_classes:int):
    """
    Создает экземпляр генератора BigGAN с заданной конфигурацией.

    Возвращает:
    - generator: Экземпляр генератора BigGAN.
    """
    
    # Создание конфигурации BigGAN
    conf = BigGANConfig()
    conf = conf.from_dict({'output_dim': 512,
                            'z_dim': 128,
                            'class_embed_dim': 128,
                            'channel_width': 128,
                            'num_classes': num_classes,
                            'layers': [[False, 16, 16],
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
                            'attention_layer_position': 8,
                            'eps': 0.0001,
                            'n_stats': 51})

    # Создание экземпляра генератора BigGAN с заданной конфигурацией
    return BigGAN(conf)

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def load_generator(num_classes, ckp):
    """
    Загружает состояние генератора из чекпоинта.

    Параметры:
    - ckp: Путь к файлу чекпоинта или словарь состояния генератора.

    Возвращает:
    - generator: Экземпляр генератора BigGAN с загруженным состоянием.
    """
    
    # Создание экземпляра генератора
    generator = instance_generator(num_classes)
    
    # Загрузка состояния генератора из чекпоинта
    if isinstance(ckp, str):
        generator.load_state_dict(torch.load(ckp))
    else:
        generator.load_state_dict(ckp)
        
    return generator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def setup_generator(num_classes, unfreeze_last_n, ckp=None):
    """
    Настраивает генератор, загружая его состояние из чекпоинта или используя предобученную модель.

    Параметры:
    - num_classes:     Количество классов
    - unfreeze_last_n: Сколько последних слоев генератора разморозить
    - ckp:             Путь к файлу чекпоинта или словарь состояния генератора (опционально).

    Возвращает:
    - generator: Экземпляр генератора BigGAN с загруженным или предобученным состоянием.
    """
    
    # Если чекпоинт предоставлен, загружаем состояние генератора из него
    if ckp is not None:
        generator = load_generator(num_classes, ckp['generator_state_dict'])
    else:
        # Иначе используем предобученную модель BigGAN
        generator = BigGAN.from_pretrained('biggan-deep-512')
        
    # Замораживаем большинство слоев генератора, кроме слоев генератора
    freeze_generator(generator, unfreeze_last_n)
        
    return generator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def freeze_generator(generator, unfreeze_last_n=4, unfreeze_embeddings=True):
    """
    Замораживает все слои генератора, кроме определенного количества последних слоев и слоя SelfAttn.
    
    Параметры:
    - generator: Экземпляр генератора BigGAN.
    - unfreeze_last_n: Количество последних слоев GenBlock, которые нужно разморозить.
    - unfreeze_embeddings: Разморозить ли слой эмбеддингов.
    """
    
    # Замораживаем все параметры генератора
    for param in generator.parameters():
        param.requires_grad = False

    # Размораживаем слой эмбеддингов при необходимости
    if unfreeze_embeddings:
        for param in generator.embeddings.parameters():
            param.requires_grad = True

    # Индекс слоя SelfAttn
    attn_index = 8

    # Размораживаем SelfAttn слой
    for param in generator.generator.layers[attn_index].parameters():
        param.requires_grad = True

    # Размораживаем последние unfreeze_last_n слоев GenBlock
    num_blocks = len(generator.generator.layers)
    unfreeze_start_index = max(0, num_blocks - unfreeze_last_n)
    for i in range(unfreeze_start_index, num_blocks):
        if isinstance(generator.generator.layers[i], GenBlock):
            for param in generator.generator.layers[i].parameters():
                param.requires_grad = True

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def setup_discriminator(num_classes, ckp=None):
    """
    Создает экземпляр дискриминатора и инициализирует его веса.

    Возвращает:
    - discriminator: Экземпляр дискриминатора.
    """
    
    # Создание экземпляра дискриминатора
    discriminator = Discriminator(512, num_classes)
    
    # Инициализация весов дискриминатора
    discriminator.apply(initialize_weights)
    
    if ckp is not None:
        discriminator.load_state_dict(ckp['discriminator_state_dict'])
    
    return discriminator

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def initialize_weights(m):
    """
    Инициализирует веса слоев модели.

    Параметры:
    - m: Слой модели.
    """
    
    # Инициализация весов Conv2d слоев с использованием Kaiming normal
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    # Инициализация весов и смещений BatchNorm2d слоев
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    # Инициализация весов Linear слоев
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)