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

# ----------------------------
# Setup 3 models
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Stacked denoising AE (SDAE): trained with k>1 re-feeds on noisy inputs
sdae = DenoisingAutoencoder().to(device)

# 2) Single-pass denoising AE (DAE): trained 1 pass on noisy inputs
dae  = DenoisingAutoencoder().to(device)

# 3) Plain AE: trained 1 pass on clean inputs
ae   = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()
opt_sdae = optim.Adam(sdae.parameters(), lr=1e-3)
opt_dae  = optim.Adam(dae.parameters(),  lr=1e-3)
opt_ae   = optim.Adam(ae.parameters(),   lr=1e-3)

noise_factor = 0.81
epochs = 10
stack_k_train = 3     # SDAE training re-feeds
stack_k_eval  = 3     # SDAE default eval re-feeds

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(1, epochs + 1):
    sdae.train(); dae.train(); ae.train()
    train_sum = {"sdae":0.0, "dae":0.0, "ae":0.0}
    train_cnt = 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        noisy = add_noise(imgs, noise_factor)

        bsz = imgs.size(0)

        # ---- SDAE: noisy -> clean with k re-feeds
        x = forward_k(sdae, noisy, stack_k_train)
        loss_sdae = criterion(x, imgs)
        opt_sdae.zero_grad(set_to_none=True)
        loss_sdae.backward()
        opt_sdae.step()

        # ---- DAE: noisy -> clean with 1 pass
        out_dae = dae(noisy)
        loss_dae = criterion(out_dae, imgs)
        opt_dae.zero_grad(set_to_none=True)
        loss_dae.backward()
        opt_dae.step()

        # ---- AE: clean -> clean with 1 pass
        out_ae = ae(imgs)
        loss_ae = criterion(out_ae, imgs)
        opt_ae.zero_grad(set_to_none=True)
        loss_ae.backward()
        opt_ae.step()

        train_sum["sdae"] += loss_sdae.item() * bsz
        train_sum["dae"]  += loss_dae.item()  * bsz
        train_sum["ae"]   += loss_ae.item()   * bsz
        train_cnt         += bsz

    # ---- Eval in natural regimes
    sdae_test = evaluate(sdae, test_loader, denoising=True,  re_feeds=stack_k_eval, noise_factor=noise_factor, device=device, criterion=criterion)
    dae_test  = evaluate(dae,  test_loader, denoising=True,  re_feeds=1,               noise_factor=noise_factor, device=device, criterion=criterion)
    ae_test   = evaluate(ae,   test_loader, denoising=False, re_feeds=1,               noise_factor=noise_factor, device=device, criterion=criterion)

    print(
        f"Epoch {epoch:02d} | "
        f"SDAE train_MSE={train_sum['sdae']/train_cnt:.4f} test_MSE={sdae_test:.4f} (k_train={stack_k_train}, k_eval={stack_k_eval}) | "
        f"DAE  train_MSE={train_sum['dae']/train_cnt:.4f}  test_MSE={dae_test:.4f} | "
        f"AE   train_MSE={train_sum['ae']/train_cnt:.4f}   test_MSE={ae_test:.4f}"
    )

# ----------------------------
# Iterative re-feed sweep for SDAE (k=1..100), with baselines
# ----------------------------
sdae.eval(); dae.eval(); ae.eval()

@torch.no_grad()
def sdae_k_sweep_mse(model: nn.Module,
                     loader,
                     ks: List[int],
                     noise_factor: float,
                     device: torch.device,
                     criterion = nn.MSELoss()) -> Dict[int, float]:
    """
    Compute test MSE for each k re-feeds on the *same* test set, denoising mode.
    Returns dict {k: mse}.
    """
    results = {}
    for k in ks:
        mse_k = evaluate(model, loader, denoising=True, re_feeds=k,
                         noise_factor=noise_factor, device=device, criterion=criterion)
        results[k] = mse_k
    return results

# Baseline reference (single numbers)
dae_mse  = evaluate(dae, test_loader, denoising=True,  re_feeds=1, noise_factor=noise_factor, device=device, criterion=criterion)
ae_mse   = evaluate(ae,  test_loader, denoising=False, re_feeds=1, noise_factor=noise_factor, device=device, criterion=criterion)

ks = list(range(1, 101))
sdae_curve = sdae_k_sweep_mse(sdae, test_loader, ks, noise_factor, device, criterion)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(9, 4.5))
plt.plot(ks, [sdae_curve[k] for k in ks], marker='o', linestyle='-', label='SDAE (k re-feeds)')
plt.axhline(dae_mse, color='gray', linestyle='--', label=f'DAE (1 pass): {dae_mse:.4f}')
plt.axhline(ae_mse,  color='black', linestyle=':',  label=f'AE (clean→clean): {ae_mse:.4f}')
plt.xlabel('Number of Re-feeds (k)')
plt.ylabel('Test Reconstruction MSE vs. Clean')
plt.title('SDAE Re-feed Depth vs. Test MSE (MNIST)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------
# Optional: quick sample visualization of denoising
# ----------------------------
@torch.no_grad()
def visualize_examples(model_sdae, model_dae, model_ae, loader, k_sdae=3, noise_factor=0.81, num_show=8):
    imgs, _ = next(iter(loader))
    imgs = imgs[:num_show].to(device)
    noisy = add_noise(imgs, noise_factor)

    deno_sdae = forward_k(model_sdae, noisy, k_sdae)
    deno_dae  = model_dae(noisy)
    recon_ae  = model_ae(imgs)

    # grid: clean | noisy | SDAE | DAE | AE (for reference)
    grid = torch.cat([imgs.cpu(), noisy.cpu(), deno_sdae.cpu(), deno_dae.cpu(), recon_ae.cpu()], dim=0)
    grid = make_grid(grid, nrow=num_show, pad_value=1.0)
    plt.figure(figsize=(num_show * 1.2, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"SDAE k={k_sdae} | DAE 1-pass | AE 1-pass")
    plt.show()

# Uncomment to visualize
# visualize_examples(sdae, dae, ae, test_loader, k_sdae=stack_k_eval, noise_factor=noise_factor, num_show=8)