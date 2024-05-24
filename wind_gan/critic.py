import torch
import torch.nn as nn
from wind_gan.dft_layer import DFTLayer


# Discriminateur utilisant la DFTLayer
class Critic(nn.Module):
    def __init__(self, signal_length):
        super(Critic, self).__init__()
        self.dft_layer = DFTLayer(signal_length)
        self.model = nn.Sequential(
            nn.Linear(signal_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        real_part, imag_part = self.dft_layer(x)
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        magnitude = magnitude.view(magnitude.size(0), -1)  # Flatten
        return self.model(magnitude)
