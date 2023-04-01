import torchvision.transforms as T


EPOCHS = 100

LEARNING_RATE = 1E-3

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.Grayscale(),
])
