import torch
import torch.nn as nn

class InceptionPerceptualLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(InceptionPerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inception, fake, real):
        with torch.no_grad():
            fake_pred = inception(fake)[0]
            real_pred = inception(real)[0]

        fake_pred = fake_pred.squeeze(3).squeeze(2)
        real_pred = real_pred.squeeze(3).squeeze(2)
        
        loss = self.mse(fake_pred, real_pred)
            
        return loss
    
class DiscriminatorHingeLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(DiscriminatorHingeLoss, self).__init__()

    def forward(self, real_output, fake_output):
        real_loss = torch.mean(torch.relu(1.0 - real_output))  # Max(0, 1 - real_output)
        fake_loss = torch.mean(torch.relu(1.0 + fake_output))  # Max(0, 1 + fake_output)
        
        return real_loss + fake_loss

class GeneratorHingeLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(GeneratorHingeLoss, self).__init__()

    def forward(self, fake_output):
        # Hinge loss for generator: Max(0, -fake_output)
        loss = -torch.mean(fake_output)
        return loss