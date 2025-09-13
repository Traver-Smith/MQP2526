import torch
from torchvision import datasets, transforms
def add_noise(img, noise_factor=0.81):
    """
    Add Gaussian noise to an input image tensor in [0,1]
    """
    noisy= img + noise_factor * torch.rand_like(img)
    return torch.clamp(noisy, 0., 1.)

