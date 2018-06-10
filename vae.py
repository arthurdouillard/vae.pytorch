import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1 * 28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 120),
            nn.ReLU(True)
        )

        self.z_mean = nn.Linear(120, 30)
        self.z_log_var = nn.Linear(120, 30)

        self.decoder = nn.Sequential(
            nn.Linear(30, 120),
            nn.ReLU(True),
            nn.Linear(120, 500),
            nn.ReLU(True),
            nn.Linear(500, 1 * 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        if self.training:
            z_std = torch.exp(0.5 * z_log_var)
            eps = torch.randn_like(z_std)
            z = eps * z_std + z_mean
        else:
            z = z_mean

        return self.decoder(z), z_mean, z_log_var
