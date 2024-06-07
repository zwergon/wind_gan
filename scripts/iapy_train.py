import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad

from iapytoo.train.wgan import WGAN
from iapytoo.utils.config import Config

from wind_gan.generator import CNN1DGenerator
from wind_gan.critic import CNN1DDiscriminator
from wind_gan.dataset import BinDataset, LatentDataset

from iapytoo.train.factories import ModelFactory, OptimizerFactory
from iapytoo.predictions.plotters import FakePlotter


if __name__ == "__main__":
    config = Config(os.path.join(os.path.dirname(__file__), "config_wgan.json"))

    # load training data
    trainset = BinDataset()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    latentset = LatentDataset(config.noise_dim, size=16)

    valid_loader = torch.utils.data.DataLoader(latentset, batch_size=16, shuffle=False)

    model_factory = ModelFactory()
    model_factory.register_model("generator", CNN1DGenerator)
    model_factory.register_model("critic", CNN1DDiscriminator)

    wgan = WGAN(config, prediction_plotter=FakePlotter(n_plot=2))
    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=None)
