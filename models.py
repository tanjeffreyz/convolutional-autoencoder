import math
import torch
import torch.nn as nn


class SimpleAutoencoder:
    def __init__(self, in_size):
        assert in_size % 2 == 0, 'Dimensions must be divisible by 2'

        self.in_size = in_size
        size = in_size * in_size
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size, size // 2),
            nn.Linear(size // 2, size)
        )

    def __call__(self, x):
        result = self.model.forward(x)
        return torch.reshape(result, (result.shape[0], 1, self.in_size, self.in_size))


class ConvolutionalAutoencoder:
    def __init__(self, _):      # in_size doesn't matter for CAE
        self.model = nn.Sequential(
            # Encoding
            nn.Conv2d(1, 2, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),

            # Latent Space
            nn.ConvTranspose2d(8, 8, kernel_size=5, stride=2, padding=2),

            # Decoding
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2, padding=1),
            nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1),
        )

    def __call__(self, x):
        return self.model.forward(x)
