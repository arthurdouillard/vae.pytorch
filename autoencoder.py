import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, dims):
        super(AutoEncoder, self).__init__()

        self.encoder = self._build_seq(dims)
        self.decoder = self._build_seq(dims[::-1], decode=True)


    def _build_seq(self, dims, decode=False):
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(True) if not decode or i != len(dims) - 2 else nn.Sigmoid()
            ])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
