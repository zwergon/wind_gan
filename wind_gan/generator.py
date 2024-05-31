import torch
import torch.nn as nn

from iapytoo.train.factories import Model


# Générateur
class Generator(Model):
    def __init__(self, loader, config) -> None:
        super(Generator, self).__init__(loader, config)
        noise_dim = config["noise_dim"]
        dataset = loader.dataset

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dataset.signal_length),
        )

    def forward(self, z):
        return self.model(z)


# Définir le générateur
class ARGenerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, ar_size=7):
        super(ARGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.ar_size = ar_size
        self.ar_coefficient = nn.Parameter(
            torch.randn(self.ar_size) / self.ar_size
        )  # Coefficient AR pour le bruit, Attention à la divergence.
        self.output_size = output_size

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        # Générer le bruit AR
        noise = self.generate_ar_noise(x.size(0), x.size(1))

        return torch.add(x, noise)  # Ajouter le bruit AR au signal généré

    def generate_ar_noise(self, batch_size, sequence_length):
        noise = torch.randn(batch_size, sequence_length)  # Générer un bruit gaussien
        ar_noise = torch.zeros_like(noise)

        # Calculer le bruit AR en utilisant le coefficient AR
        for t in range(self.ar_size, sequence_length):
            ar_noise[:, t] = noise[:, t]
            for i in range(self.ar_size):
                ar_noise[:, t] += ar_noise[:, t - i - 1] * self.ar_coefficient[i]

        return ar_noise
