import unittest
import torch
from wind_gan.generator import ARGenerator, CNN1DGenerator, GruGenerator
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
        noise = CNN1DGenerator.get_noise(config["batch_size"], config["noise_dim"])
        print(noise.shape)
        out = netG(noise)
        print(out.shape)

    def test_gru(self):
        config = {"batch_size": 6, "noise_dim": 100, "hidden_size": 128}

        dataset = BinDataset()
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )

        netG = GruGenerator(config=config, loader=trainloader)
        noise = netG.get_noise(config["batch_size"], config["noise_dim"])
        self.assertListEqual(list(noise.shape), [6, 100])
        out = netG(noise)
        
        self.assertListEqual(list(out.shape), [6, 1, 1200])


if __name__ == "__main__":
    unittest.main()
