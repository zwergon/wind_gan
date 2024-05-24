import os
import unittest
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from wind_gan.dft_layer import DFTLayer

input_size = 256
model_name = os.path.join(os.path.dirname(__file__), "data/dft_model.pth")


# Définition du modèle
class DFTNet(nn.Module):
    def __init__(self, input_size):
        super(DFTNet, self).__init__()
        self.dft_layer = DFTLayer(input_size)

    def forward(self, x):
        real_part, imag_part = self.dft_layer(x)
        return real_part, imag_part


# Génération de données d'entraînement
def generate_data(num_samples, signal_length):
    X = np.random.rand(num_samples, signal_length)
    Y = np.fft.fft(X, axis=1)
    Y_real = Y.real
    Y_imag = Y.imag
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y_real, dtype=torch.float32),
        torch.tensor(Y_imag, dtype=torch.float32),
    )


def train_model():
    # Paramètres

    num_samples = 10000
    num_epochs = 1000
    learning_rate = 0.001

    # Données d'entraînement
    X_train, Y_train_real, Y_train_imag = generate_data(num_samples, input_size)

    # Initialisation du modèle, de la perte et de l'optimiseur
    model = DFTNet(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs_real, outputs_imag = model(X_train)
        loss_real = criterion(outputs_real, Y_train_real)
        loss_imag = criterion(outputs_imag, Y_train_imag)
        loss = loss_real + loss_imag
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save({"model": model.state_dict()}, model_name)


class TestDFTLayer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_model(self):
        state_dict = torch.load(model_name)
        model = DFTNet(input_size=input_size)
        model.load_state_dict(state_dict["model"])

        # Test du modèle sur un nouvel échantillon
        model.eval()
        with torch.no_grad():
            test_signal = torch.tensor(
                np.random.rand(1, input_size), dtype=torch.float32
            )
            pred_real, pred_imag = model(test_signal)
            pred_complex = pred_real.numpy() + 1j * pred_imag.numpy()
            true_complex = np.fft.fft(test_signal.numpy())

            pred_modulus = np.sqrt(pred_complex.real**2 + pred_complex.imag**2)
            true_modulus = np.sqrt(true_complex.real**2 + true_complex.imag**2)

            print("Predicted DFT:", pred_complex)
            print("True DFT:", true_complex)

            plt.scatter(pred_modulus, true_modulus)
            plt.show()


if __name__ == "__main__":
    train_model()
    unittest.main()
