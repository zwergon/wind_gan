import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad

from iapytoo.train.wgan import WGAN
from iapytoo.utils.config import Config

from wind_gan.generator import Generator
from wind_gan.critic import Critic
from wind_gan.dataset import BinDataset

from iapytoo.train.factories import ModelFactory, OptimizerFactory


if __name__ == "__main__":
    config = Config(os.path.join(os.path.dirname(__file__), "config_wgan.json"))

    # load training data
    trainset = BinDataset()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    model_factory = ModelFactory()
    model_factory.register_model("generator", Generator)
    model_factory.register_model("critic", Critic)

    o_factory = OptimizerFactory()
    print(o_factory.optimizers_dict)

    wgan = WGAN(config)
    wgan.fit(train_loader=trainloader, valid_loader=None, run_id=None)
