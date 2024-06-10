import unittest
import numpy as np
import matplotlib.pyplot as plt
from wind_gan.dataset import BinDataset, SinDataset


class TestDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_bin_dataset(self):
        dataset = BinDataset()

        print("signal_length", dataset.signal_length)
        print("dataset size", len(dataset))
        for i in range(3):
            idx = int(np.random.random() * len(dataset))
            print("idx", idx)
            spectra, target = dataset[idx]
            plt.plot(spectra[0, :] + i)
        plt.show()

    def test_sindataset(self):
        dataset = SinDataset()
        print("signal_length", dataset.signal_length)
        print("dataset size", len(dataset))
        for i in range(3):
            idx = int(np.random.random() * len(dataset))
            signal, labels = dataset[idx]
            print(f"idx {idx} labels: {labels}")
            signal = signal.flatten()
            plt.plot(signal + i)
        plt.show()


if __name__ == "__main__":
    unittest.main()
