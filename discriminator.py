import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class Discriminator(nn.Module):
    def __init__(self, resolution, num_classes, dropout_prob=0.3):
        super(Discriminator, self).__init__()
        
        self.resolution = resolution
        
        # Слой эмбеддингов классов
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

            # Дополнительные сверточные слои
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            spectral_norm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            spectral_norm(nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=1))  # Выходной слой с 1 каналом
        )
        
        self.ff = nn.Linear((resolution // 32), 1)
        
        self.classifier = nn.Sequential(
            nn.Linear((resolution // 32), num_classes),
            nn.Softmax(dim=1)  # Для многоклассовой классификации
        )

    def forward(self, img, labels):
        # Преобразуем метки классов в эмбеддинги
        embeddings = self.embedding(labels)
        embeddings = embeddings.view(labels.size(0), 1, self.resolution, self.resolution)
        
        # Конкатенируем изображение и эмбеддинги классов по каналу
        x = torch.cat([img, embeddings], dim=1)
        
        x = self.model(x)
        
        x = x.view(x.size(0), -1)
        
        binar_output = self.ff(x)
        class_output = self.classifier(x)
        
        return binar_output, class_output