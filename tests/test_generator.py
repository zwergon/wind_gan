import unittest
import torch
from wind_gan.generator import ARGenerator, CNN1DGenerator
from wind_gan.dataset import BinDataset


class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_generator(self):
        # Paramètres du GAN
        input_size = 100  # Taille de l'entrée aléatoire pour le générateur
        # hidden_size = 128  # Taille des couches cachées
        output_size = 1000  # Taille de la séquence de sortie (signal 1D)

        torch.manual_seed(0)

        # Instancier le générateur
        generator = ARGenerator(input_size, output_size)

        # Exemple d'utilisation
        random_input = torch.randn(input_size).view(
            1, -1
        )  # Générer une entrée aléatoire avec un batch de taille 1
        generated_signal = generator(random_input)  # Générer un signal 1D avec bruit AR
        print(generated_signal)

    def test_cnn1d(self):
        config = {"batch_size": 6, "noise_dim": 100}

        dataset = BinDataset()
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )

        netG = CNN1DGenerator(config=config, loader=trainloader)
        noise = torch.randn(config["batch_size"], config["noise_dim"], 1)
        print(noise.shape)
        out = netG(noise)
        print(out.shape)


if __name__ == "__main__":
    unittest.main()
