import os
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


def show_test_images(path):
    kind = path.split('/')[1]
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
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    x = next(iter(test_loader))[0]
    if kind == 'ConvolutionalAutoencoder':
        model_class = ConvolutionalAutoencoder
    else:
        model_class = SimpleAutoencoder

    model = model_class(x.shape[-1]).to(device)
    weights = torch.load(os.path.join(path, 'weights', 'final'))
    model.load_state_dict(weights)
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            predictions = model.forward(data)

            data_grid = make_grid(data)[0]
            pred_grid = make_grid(predictions)[0]
            fig, axs = plt.subplots(ncols=2, squeeze=False, figsize=(10, 10))

            # Plot ground truth
            _plot_grid(axs[0, 0], data_grid)

            # Plot predictions
            _plot_grid(axs[0, 1], pred_grid)

            plt.show()
            del data


SAE_PATH = 'models/SimpleAutoencoder/03_24_2023/18_06_04'
CAE_PATH = 'models/ConvolutionalAutoencoder/03_24_2023/18_21_49'

# plot_losses(SAE_PATH, 'orange')
# plot_losses(CAE_PATH, 'blue')
# show_test_images(SAE_PATH)
show_test_images(CAE_PATH)

plt.show()
