import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_losses(path, weights, color):
    model_class = path.split('/')[1]
    test_loss_path = os.path.join(path, 'test_losses.npy')
    train_loss_path = os.path.join(path, 'train_losses.npy')

    test_losses = np.load(test_loss_path)
    train_losses = np.load(train_loss_path)

    sns.lineplot(x=train_losses[0, :], y=train_losses[1, :], color=color, alpha=0.25)
    sns.lineplot(x=test_losses[0, :], y=test_losses[1, :], color=color, label=model_class)


plot_losses('models/SimpleAutoencoder/03_24_2023/18_06_04', 'final', 'orange')
plot_losses('models/ConvolutionalAutoencoder/03_24_2023/18_21_49', 'final', 'blue')
plt.show()
