import torch
import torch.nn as nn
from wind_gan.dft_layer import DFTLayer
from iapytoo.train.factories import Model


# Discriminateur utilisant la DFTLayer
class Critic(Model):
    def __init__(self, loader, config) -> None:
        super().__init__(loader, config)
        dataset = loader.dataset
        self.dft_layer = DFTLayer(dataset.signal_length)
        self.model = nn.Sequential(
            nn.Linear(dataset.signal_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def to(self, device):
        self.dft_layer = self.dft_layer.to(device)
        return super().to(device)

    def forward(self, x):
        real_part, imag_part = self.dft_layer(x)
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        magnitude = magnitude.view(magnitude.size(0), -1)  # Flatten
        return self.model(magnitude)
