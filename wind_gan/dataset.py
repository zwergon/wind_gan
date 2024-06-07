import os
import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt


class _Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.dataset = np.loadtxt(filename, delimiter=",").astype(np.float32)
        # self.minmax_normalize()

    def __len__(self):
        return self.dataset.shape[0]

    @property
    def signal_length(self):
        return self.dataset.shape[1] - 1  # last column is one index

    def __getitem__(self, idx):
        step = self.dataset[idx : idx + 1, :-1]  # add channel [1, sequence_length]
        target = self.dataset[idx, -1]
        return step, target


class BinDataset(_Dataset):
    def __init__(self):
        super().__init__(
            filename=os.path.join(
                os.path.dirname(__file__),
                "../tests/data/bin.csv",
            )
        )


class LatentDataset(Dataset):
    def __init__(self, noise_dim, size=1) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        noise = torch.randn(self.noise_dim, 1)
        return noise


if __name__ == "__main__":
    dataset = BinDataset()

    print("signal_length", dataset.signal_length)
    print("dataset size", len(dataset))
    for i in range(3):
        idx = int(np.random.random() * len(dataset))
        print("idx", idx)
        spectra, target = dataset[idx]
        plt.plot(spectra + i)
    plt.show()
