import torch
from torch import nn

class AutoEncoderConv(nn.Module):
    def __init__(self, in_channels):
        super(AutoEncoderConv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 3, 5),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(3, 12, 5),
            nn.ReLU(True),
            nn.AvgPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 3, 10),
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 1, 16),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
