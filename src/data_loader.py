import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml

def get_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_mnist_dataloaders(config):
    """
    Returns train, validation, and test DataLoaders for MNIST
    """
    # Config params
    batch_size = config['dataset']['batch_size']
    val_split = config['dataset']['validation_split']
    download = config['dataset']['download']
    shuffle = config['dataset']['shuffle']
    path = config['dataset']['path']

    # 1. Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Load full training dataset
    full_train = datasets.MNIST(root=path, train=True, transform=transform, download=download)

    # 3. Create train/validation split
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    # 4. Test dataset
    test_dataset = datasets.MNIST(root=path, train=False, transform=transform, download=download)

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
