import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, resolution, num_classes, dropout_prob=0.3):
        super(Discriminator, self).__init__()

        self.resolution = resolution
        self.embedding = nn.Embedding(num_classes, resolution * resolution)

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1))
        )

        # Adjust Linear layers for the final feature map size (16x16)
        self.fc_bin = nn.Linear(16 * 16, 1)
        self.fc_class = nn.Linear(16 * 16, num_classes)

    def forward(self, img, labels):
        embeddings = self.embedding(labels).view(labels.size(0), 1, self.resolution, self.resolution)
        x = torch.cat([img, embeddings], dim=1)
        x = self.model(x).view(x.size(0), -1)  # Flatten 16x16 feature map
        binar_output = self.fc_bin(x)
        class_output = self.fc_class(x)

        return binar_output, class_output
