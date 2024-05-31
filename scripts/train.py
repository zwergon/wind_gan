import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad

from wind_gan.generator import Generator
from wind_gan.critic import Critic
from wind_gan.dataset import BinDataset


# Fonction pour la pénalité de gradient
def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1).expand_as(real_data)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == "__main__":
    # Paramètres

    noise_dim = 100
    batch_size = 64
    num_epochs = 10000
    learning_rate = 0.001
    lambda_gp = 10

    # load training data
    trainset = BinDataset()

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Initialisation des modèles
    generator = Generator(noise_dim, trainset.signal_length)
    discriminator = Critic(trainset.signal_length)

    # Optimiseurs
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
    )

    # Entraînement du WGAN
    for epoch in range(num_epochs):
        for step, (real_data, _) in enumerate(trainloader):
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise).detach()

            optimizer_D.zero_grad()

            # Pertes pour les vrais et faux
            d_real = discriminator(real_data)
            d_fake = discriminator(fake_data)

            # Pénalité de gradient
            gp = gradient_penalty(discriminator, real_data, fake_data)

            # Perte totale du discriminateur
            d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()

        # Mise à jour du générateur
        noise = torch.randn(batch_size, noise_dim)
        fake_data = generator(noise)

        optimizer_G.zero_grad()

        g_loss = -discriminator(fake_data).mean()
        g_loss.backward()
        optimizer_G.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
            )

    # Sauvegarde des modèles et des optimisateurs
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
        },
        "wgan_dft.pth",
    )

    # Test du générateur après entraînement
    with torch.no_grad():
        noise = torch.randn(1, noise_dim)
        generated_signal = generator(noise)
        print("Generated Signal:", generated_signal)
