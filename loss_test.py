import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Dict, List

# ----------------------------
# Noise
# ----------------------------
def add_noise(img, noise_factor=0.81):
    """
    Add Gaussian noise to an input image tensor in [0,1], then clamp.
    """
    noisy = img + noise_factor * torch.randn_like(img)
    return torch.clamp(noisy, 0., 1.)
# ----------------------------
# Data
# ----------------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# ----------------------------
# Model
# ----------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ----------------------------
# Utilities
# ----------------------------
def forward_k(model: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Feed 'x' through 'model' k times (re-feeds).
    """
    for _ in range(k):
        x = model(x)
    return x

@torch.no_grad()
def batch_mse(a, b):
    return ((a - b) ** 2).mean().item()

def evaluate(model: nn.Module,
             loader,
             *,
             denoising: bool,
             re_feeds: int = 1,
             noise_factor: float = 0.81,
             device: torch.device = torch.device("cpu"),
             criterion = nn.MSELoss()) -> float:
    """
    Evaluate reconstruction MSE against the clean target images.
    - If denoising=True, inputs = add_noise(clean), targets = clean.
    - If denoising=False, inputs = clean, targets = clean.
    - The model output is forward_k(model, inputs, re_feeds).
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            inputs = add_noise(imgs, noise_factor) if denoising else imgs
            recon = forward_k(model, inputs, re_feeds)
            loss = criterion(recon, imgs)
            bsz = imgs.size(0)
            total += loss.item() * bsz
            count += bsz
    return total / max(count, 1)
