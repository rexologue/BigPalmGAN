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
