# data.py
# Data loaders for MNIST and FashionMNIST.
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loader(dataset: str = "mnist", batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, int]:
    dataset = dataset.lower()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # scale to [-1, 1]
    ])

    if dataset == "mnist":
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        n_classes = 10
    elif dataset == "fashion":
        ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
        n_classes = 10
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader, n_classes
