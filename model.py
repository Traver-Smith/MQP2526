import torch
import torch.nn as nn
from torchvision import datasets, transforms
from .DenoisingAutoencoder import DAE
from .performance import evaluate, plot_loss_curves, plot_refeed_curves
from .data_processing import add_noise
from .stacking_utils import  stacked_dae_loss
import random
# ----------------------------
# Data
# ----------------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

print('loading')
history = {
    "AE": {"train": [], "val": []},
    "DAE": {"train": [], "val": []},
    "sDAE": {"train": [], "val_final": [], "val_multi": []}
}

# ----------------------------
# Setup 3 models
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Stacked denoising AE (SDAE): trained with k>1 re-feeds on noisy inputs
sdae = DAE().to(device)

# 2) Single-pass denoising AE (DAE): trained 1 pass on noisy inputs
dae  = DAE().to(device)

# 3) Plain AE: trained 1 pass on clean inputs
ae   = DAE().to(device)

num_epochs = 20
lr = 1e-3
criterion = nn.MSELoss()
opt_ae   = torch.optim.Adam(ae.parameters(), lr=lr)
opt_dae  = torch.optim.Adam(dae.parameters(), lr=lr)
opt_sdae = torch.optim.Adam(sdae.parameters(), lr=lr)

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(1, num_epochs + 1):
    ae.train()
    total_loss, count = 0.0, 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        opt_ae.zero_grad()

        #Forward pass (clean -> clean, 1 feed)
        recon = ae(imgs)
        loss = criterion(recon, imgs)

        #Backwards + update
        loss.backward()
        opt_ae.step()

        # Accumulate training loss
        bsz = imgs.size(0)
        total_loss += loss.item() * bsz
        count += bsz
    
    #Average training loss
    train_loss = total_loss / max(count, 1)

    #Evaluate on test set (clean -> clean)
    val_loss = evaluate(ae, test_loader, device=device, criterion=criterion)
    history["AE"]["train"].append(train_loss)
    history["AE"]["val"].append(val_loss)

    print(f"Epoch [{epoch:03d}/{num_epochs}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# -------------------------------------
# Non-stacked DAE Training Loop
# --------------------------------------

for epoch in range(1, num_epochs + 1):
    dae.train()
    total_loss, count = 0.0, 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        noisy_imgs = add_noise(imgs, noise_factor=0.81)

        opt_dae.zero_grad()
        recon = dae(noisy_imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        opt_dae.step()

        bsz = imgs.size(0)
        total_loss += loss.item() * bsz
        count += bsz

    train_loss = total_loss / max(count, 1)

    # use general evaluate (denoising=True, re_feeds=1, stacked=False)
    val_loss = evaluate(dae, test_loader,
                        denoising=True,
                        re_feeds=1,
                        stacked=False,
                        device=device,
                        criterion=criterion)
    history["DAE"]["train"].append(train_loss)
    history["DAE"]["val"].append(val_loss)

    print(f"[DAE] Epoch [{epoch:03d}/{num_epochs}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# -------------------------------------
# Stacked DAE Training Loop
# --------------------------------------
k_max = 5
for epoch in range(1, num_epochs + 1):
    sdae.train()
    total_final_loss, count = 0.0, 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        noisy_imgs = add_noise(imgs, noise_factor=0.81)

        k_refeeds = random.randint(1, k_max)

        opt_sdae.zero_grad()
        loss, final_out = stacked_dae_loss(sdae, noisy_imgs, imgs, k=k_refeeds)
        loss.backward()
        opt_sdae.step()

        # log *final-pass* train loss for comparability
        final_loss = torch.mean((final_out - imgs) ** 2).item()
        bsz = imgs.size(0)
        total_final_loss += final_loss * bsz
        count += bsz

    train_loss_final = total_final_loss / max(count, 1)

    # Validation: log both final and multi-step
    val_loss_final = evaluate(sdae, test_loader,
                              denoising=True,
                              re_feeds=k_max,
                              stacked=True,
                              multi_step=False,
                              device=device,
                              criterion=criterion)

    val_loss_multi = evaluate(sdae, test_loader,
                              denoising=True,
                              re_feeds=k_max,
                              stacked=True,
                              multi_step=True,
                              device=device,
                              criterion=criterion)

    history["sDAE"]["train"].append(train_loss_final)   # comparable to AE/DAE
    history["sDAE"]["val_final"].append(val_loss_final) # comparable metric
    history["sDAE"]["val_multi"].append(val_loss_multi) # diagnostic

    print(f"[Stacked DAE] Epoch [{epoch:03d}/{num_epochs}] "
          f"Train Final: {train_loss_final:.6f} | "
          f"Val Final: {val_loss_final:.6f} | Val Multi: {val_loss_multi:.6f}")

plot_loss_curves(history, num_epochs, show_sdae_multi=True)

models = {"AE": ae, "DAE": dae, "sDAE": sdae}
plot_refeed_curves(models, test_loader, device, max_refeeds=150)