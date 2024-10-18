import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

################################################################
# //////////////////////////////////////////////////////////// #
################################################################
class ConditionalTransform:
    """
    Класс для применения условных преобразований к изображениям в зависимости от метки класса.
    """

    def __init__(self):
        """
        Инициализация словаря преобразований для каждого класса и нормализации.
        """
        # Определение преобразований для каждого класса (0, 1, 2, 3)
        self.transforms_dict = {
            0: transforms.Compose([
                transforms.RandomRotation(5),  # Меньший угол поворота
                transforms.ColorJitter(brightness=0.02, contrast=0.02),  # Более слабые изменения яркости и контрастности
                transforms.RandomResizedCrop(512, scale=(0.95, 1.0)),  # Уменьшена область ресайза
                transforms.ToTensor()
            ]),
            1: transforms.Compose([
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.04, contrast=0.04),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),  # Уменьшено смещение
                transforms.ToTensor()
            ]),
            2: transforms.Compose([
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.03, contrast=0.03),
                transforms.RandomResizedCrop(512, scale=(0.96, 1.0)),
                transforms.ToTensor()
            ]),
            3: transforms.Compose([
                transforms.RandomRotation(6),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.RandomAffine(degrees=6, translate=(0.03, 0.03)),
                transforms.ToTensor()
            ])
        }

        # Нормализация, общая для всех классов
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __call__(self, img, label):
        """
        Применяет преобразования к изображению в зависимости от метки класса.

        Параметры:
        - img: Изображение для преобразования.
        - label: Метка класса изображения.

        Возвращает:
        - img: Преобразованное изображение.
        """
        # Применение преобразований на основе метки класса
        img = self.transforms_dict[label](img)
        img = self.normalize(img)
        return img
    
    
################################################################
# //////////////////////////////////////////////////////////// #
################################################################
class PalmDataset(Dataset):
    """
    Класс для загрузки изображений ладоней с метками классов.
    """
    
    def __init__(self, root_dir, transform=None, is_validation=False):
        """
        Инициализация набора данных.

        Параметры:
        - root_dir: Корневая директория, содержащая поддиректории с изображениями для каждого класса.
        - transform: Преобразования для применения к изображениям (опционально).
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.is_validation = is_validation
        self.image_paths = []
        self.labels = []

        # Загрузка путей изображений и меток классов
        for class_id in range(4):  # 4 класса: Left White, Left Dark, Right White, Right Dark
            class_dir = os.path.join(root_dir, str(class_id))
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(class_id)

    def __len__(self):
        """
        Возвращает количество изображений в наборе данных.

        Возвращает:
        - len: Количество изображений.
        """
        
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Возвращает изображение и его метку класса по индексу.

        Параметры:
        - idx: Индекс изображения.

        Возвращает:
        - image: Преобразованное изображение.
        - label: Метка класса изображения.
        """
        
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        label = self.labels[idx]

        if self.transform:
            if self.is_validation:
                image = self.transform(image)
            else:
                image = self.transform(image, label)

        return image, label


# Instantiate the ConditionalTransform
training_transform = ConditionalTransform()

# Image transforms and dataloaders
validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalization to match the pre-trained BigGAN
])


################################################################
# //////////////////////////////////////////////////////////// #
################################################################
def setup_loader(path, batch_size, use_augs, train_phase=True):
    """
    Создает загрузчик данных для обучения или валидации.

    Параметры:
    - path: Путь к директории с данными.
    - batch_size: Размер батча.
    - train_phase: Флаг, указывающий на то, является ли это фазой обучения (по умолчанию True).

    Возвращает:
    - DataLoader: Загрузчик данных.
    """
    
    # Создание набора данных с соответствующими преобразованиями
    if train_phase and use_augs:
        dataset = PalmDataset(root_dir=path, transform=training_transform, is_validation=False)
    else:
        dataset = PalmDataset(root_dir=path, transform=validation_transform, is_validation=True)
        
    # Создание загрузчика данных
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
