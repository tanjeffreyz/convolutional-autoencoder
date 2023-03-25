import os
import math
import torch
import numpy as np
import seaborn as sns
import torchvision.transforms as T
import torchvision.transforms.functional as F
from models import SimpleAutoencoder, ConvolutionalAutoencoder
from torchvision.datasets.lfw import LFWPeople
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_losses(path, color):
    model_class = path.split('/')[1]
    test_loss_path = os.path.join(path, 'test_losses.npy')
    train_loss_path = os.path.join(path, 'train_losses.npy')

    test_losses = np.load(test_loss_path)
    train_losses = np.load(train_loss_path)

    sns.lineplot(x=train_losses[0, :], y=train_losses[1, :], color=color, alpha=0.25)
    sns.lineplot(x=test_losses[0, :], y=test_losses[1, :], color=color, label=model_class)


def _plot_grid(plot, batch):
    img = batch.detach()
    img = F.to_pil_image(img)
    plot.imshow(np.asarray(img), cmap='gray', vmin=0, vmax=255)
    plot.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_test_images(sae_path, cae_path, batch_size=16):
    test_set = LFWPeople(
        root='data',
        split='test',
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Resize((64, 64)),
            T.Grayscale(),
        ])
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    x = next(iter(test_loader))[0]

    sae_model = SimpleAutoencoder(x.shape[-1]).to(device)
    weights = torch.load(os.path.join(sae_path, 'weights', 'final'))
    sae_model.load_state_dict(weights)

    cae_model = ConvolutionalAutoencoder(x.shape[-1]).to(device)
    weights = torch.load(os.path.join(cae_path, 'weights', 'final'))
    cae_model.load_state_dict(weights)
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            sae_pred = sae_model.forward(data)
            cae_pred = cae_model.forward(data)

            n_rows = int(math.sqrt(batch_size))
            data_grid = make_grid(data, nrow=n_rows)[0]
            cae_grid = make_grid(cae_pred, nrow=n_rows)[0]
            sae_grid = make_grid(sae_pred, nrow=n_rows)[0]

            fig, axs = plt.subplots(ncols=3, squeeze=False, figsize=(10, 10))

            # Plot ground truth
            _plot_grid(axs[0, 0], data_grid)
            axs[0, 0].title.set_text('Ground Truth')

            # Plot SAE predictions
            _plot_grid(axs[0, 1], sae_grid)
            axs[0, 1].title.set_text('SimpleAutoencoder')

            # Plot CAE predictions
            _plot_grid(axs[0, 2], cae_grid)
            axs[0, 2].title.set_text('ConvolutionalAutoencoder')

            plt.subplots_adjust(wspace=0.025)
            plt.show()
            del data


SAE_PATH = 'models/SimpleAutoencoder/03_24_2023/18_06_04'
CAE_PATH = 'models/ConvolutionalAutoencoder/03_24_2023/18_21_49'

# plot_losses(SAE_PATH, 'orange')
# plot_losses(CAE_PATH, 'blue')
# plt.show()

show_test_images(SAE_PATH, CAE_PATH)
