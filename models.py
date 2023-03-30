import torch
import torch.nn as nn


class ShapeProbe(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class SimpleAutoencoder(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        assert in_size % 2 == 0, 'Dimensions must be divisible by 2'

        self.in_size = in_size
        size = in_size * in_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size, size // 2),
            nn.Linear(size // 2, size),
            nn.Sigmoid()
        )

    def forward(self, x):
        result = self.model.forward(x)
        return torch.reshape(result, (result.shape[0], 1, self.in_size, self.in_size))


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, _):      # in_size doesn't matter for CAE
        super().__init__()
        self.model = nn.Sequential(
            # Encoding
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            # Latent Space
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),

            # Decoding
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 4, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)
