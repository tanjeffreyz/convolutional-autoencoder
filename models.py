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

            # Encoder
            nn.Linear(size, size // 2),
            nn.ReLU(inplace=True),

            # Latent Space (32 x 64)
            nn.Linear(size // 2, size // 2),
            nn.ReLU(inplace=True),

            # Decoder
            nn.Linear(size // 2, size),
            nn.Sigmoid()        # Sigmoid b/c grayscale pixel values are in range [0,1]
        )

    def forward(self, x):
        result = self.model.forward(x)
        return torch.reshape(result, (result.shape[0], 1, self.in_size, self.in_size))


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, _):      # in_size doesn't matter for CAE
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            # Latent Space (32 x 8 x 8)
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Decoder
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model.forward(x)
