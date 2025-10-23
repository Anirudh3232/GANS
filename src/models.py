# models.py
# Small DCGAN-style Generator and Discriminator for 28x28 grayscale images.
import torch
import torch.nn as nn

# Latent vector (noise) size
Z_DIM = 100


class Generator(nn.Module):
    """Maps a noise vector z ~ N(0,1) to a (1,28,28) image in [-1, 1]."""

    def __init__(self, z_dim: int = Z_DIM, gf: int = 64):
        super().__init__()
        # FC to a small spatial map, then upsample with ConvTranspose2d
        self.net = nn.Sequential(
            nn.Linear(z_dim, gf * 4 * 7 * 7),
            nn.BatchNorm1d(gf * 4 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (gf * 4, 7, 7)),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(gf * 4, gf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(gf * 2),
            nn.ReLU(True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(gf * 2, gf, 4, stride=2, padding=1),
            nn.BatchNorm2d(gf),
            nn.ReLU(True),
            # to 1 channel
            nn.Conv2d(gf, 1, 3, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Binary classifier: real vs fake. Outputs a single logit per image."""

    def __init__(self, df: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 1x28x28 -> 64x14x14
            nn.Conv2d(1, df, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x14x14 -> 128x7x7
            nn.Conv2d(df, df * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(df * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(df * 2 * 7 * 7, 1),  # raw logit; use BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
