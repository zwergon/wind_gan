import unittest
import torch
from wind_gan.critic import GruDiscriminator
from wind_gan.dataset import BinDataset


class TestCritic(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)


    def test_gru(self):
        config = {"batch_size": 6, "noise_dim": 100, "hidden_size": 128}

        dataset = BinDataset()
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )

        real_data, _ = next(iter(trainloader))
        print(real_data.shape)

        netD = GruDiscriminator(config=config, loader=trainloader)
        out = netD(real_data)

        print(out.shape)
        
        # self.assertListEqual(list(out.shape), [6, 1, 1200])


if __name__ == "__main__":
    unittest.main()
