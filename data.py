import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PalmDataset(Dataset):
    """Dataset class for palm images with labels."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        for class_id in range(4):  # 4 classes
            class_dir = os.path.join(root_dir, str(class_id))
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Optionally return a placeholder image or skip this sample
            image = Image.new("RGB", (512, 512))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Image transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

validation_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def setup_loader(path, batch_size, use_augs, train_phase=True):
    """Creates a data loader for training or validation."""
    if train_phase and use_augs:
        dataset = PalmDataset(root_dir=path, transform=train_transform)
    else:
        dataset = PalmDataset(root_dir=path, transform=validation_transform)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=train_phase, num_workers=4)
