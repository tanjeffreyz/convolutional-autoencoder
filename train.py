import torch
import os
import argparse
import numpy as np
import torchvision.transforms as T
from models import SimpleAutoencoder, ConvolutionalAutoencoder
from torchvision.datasets.lfw import LFWPeople
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--conv', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
now = datetime.now()

train_set = LFWPeople(
    root='data',
    split='train',
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Resize((64, 64)),
        T.Grayscale(),
    ])
)
test_set = LFWPeople(root='data', split='test', download=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

# Initialize models
x = next(iter(train_loader))[0]
model_class = (ConvolutionalAutoencoder if args.conv else SimpleAutoencoder)
model = model_class(x.shape[-1])

#
print(x.shape)
model(x)
# print(model(x).shape)


# Create folders for this run
root = os.path.join(
    'models',
    str(model.__class__.__name__),
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
weight_dir = os.path.join(root, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)
