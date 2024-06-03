import torch
import torch.nn as nn
from wind_gan.dft_layer import DFTLayer
from iapytoo.train.factories import Model


class CNN1DDiscriminator(Model):
    def __init__(self, loader, config):
        super(CNN1DDiscriminator, self).__init__(loader, config)
        noise_dim = config["noise_dim"]
        dataset = loader.dataset

        kernel_size = dataset.signal_length // 16  # 4 couches qui multiplient par 2
        self.main = nn.Sequential(
            # input kernel_size*16
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*8
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*4
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*2
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size
            nn.Conv1d(512, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


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
