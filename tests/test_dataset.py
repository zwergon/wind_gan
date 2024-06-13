import unittest
import numpy as np
import matplotlib.pyplot as plt
from wind_gan.dataset import BinDataset, SinDataset
from scipy import signal


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
            signal = signal.flatten()
            time = np.arange(0, 60, 60/600)
            print(f"idx {idx} labels: {labels}")
           
            plt.plot(time, signal + i)
        plt.show()

    def test_signal(self):
        fs = 1e3
        N = 600
        nperseg = N // 5
        freq = 200.0
        time = np.arange(N) / fs
        x = np.sin(2*np.pi*freq*time)
        f, Pxx_den = signal.welch(x, fs, nperseg=nperseg)
        plt.semilogy(f, Pxx_den)
        plt.ylim([0.5e-3, 1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()


if __name__ == "__main__":
    unittest.main()
