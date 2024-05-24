import numpy as np
import os
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, filename):
        self.dataset = np.loadtxt(filename, delimiter=",").astype(np.float32)
        # self.minmax_normalize()

    def __len__(self):
        return self.dataset.shape[0]

    @property
    def signal_length(self):
        return self.dataset.shape[1] - 1  # last column is one index

    def __getitem__(self, idx):
        step = self.dataset[idx, :-1]
        target = self.dataset[idx, -1]
        return step, target


class BinDataset(Dataset):
    def __init__(self):
        super().__init__(
            filename=os.path.join(
                os.path.dirname(__file__),
                "../tests/data/bin.csv",
            )
        )


if __name__ == "__main__":
    dataset = BinDataset()

    print("signal_length", dataset.signal_length)

    for i in range(3):
        idx = int(np.random.random() * len(dataset))
        print("idx", idx)
        spectra, target = dataset[idx]
        plt.plot(spectra + i)
    plt.show()
