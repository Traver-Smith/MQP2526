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


# Declaring models
models = {"AE": ae, "DAE": dae, "sDAE": sdae}
plot_refeed_curves(models, test_loader, device, max_refeeds=150)


# def jvp(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     """
#     Compute J_f(x) @ v (same shape as x) using autograd.
#     x, v have shape (1, D). Returns (1, D).
#     """
#     x = x.detach().requires_grad_(True)
#     y = f(x)
#     assert y.shape == x.shape, "f must be R^D -> R^D (pointwise denoiser)."
#     dot = (y * v).sum()
#     (grad_x,) = torch.autograd.grad(dot, x, retain_graph=True, create_graph=False)
#     return grad_x


# def vjp(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#     """
#     Compute J_f(x)^T @ u (same shape as x) using autograd.
#     x, u have shape (1, D). Returns (1, D).
#     """
#     x = x.detach().requires_grad_(True)
#     y = f(x)
#     (grad_x,) = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=u, retain_graph=True, create_graph=False)
#     return grad_x


# def spectral_norm_of_jacobian(
#     f: Callable[[torch.Tensor], torch.Tensor],
#     x_star: torch.Tensor,
#     iters: int = 50,
#     device: str = "cpu",
# ) -> float:
#     """
#     Estimate ||J_f(x*)||_2 (largest singular value) via power iteration on J^T J, without
#     materializing J. If < 1, f is locally a contraction at x* (sufficient condition).
#     """
#     with torch.no_grad():
#         x = x_star.view(1, -1).to(device)

#     u = torch.randn_like(x)
#     u = u / (u.norm() + 1e-12)

#     for _ in range(iters):
#         # v = (J^T J) u  ≈  J^T (J u)
#         w = jvp(f, x, u)       # w = J u
#         v = vjp(f, x, w)       # v = J^T w
#         v_norm = v.norm() + 1e-12
#         u = v / v_norm

#     # Rayleigh quotient: ||J u||_2 is the singular value estimate
#     Ju = jvp(f, x, u)
#     return float(Ju.norm().item())

import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

def detect_fixed_point(traj: torch.Tensor, eps: float = 1e-4) -> Optional[int]:
    """
    Detect first t with ||x_{t+1} - x_t||_2 < eps (approx fixed point).
    Returns the index t (0-based), or None if not found.
    """
    diffs = torch.norm(traj[1:] - traj[:-1], dim=1)
    idx = (diffs < eps).nonzero(as_tuple=False)
    return int(idx[0].item()) if idx.numel() > 0 else None


def detect_cycle(
    traj: torch.Tensor,
    max_period: int = 10,
    eps: float = 1e-3,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Search for short limit cycles by checking ||x_t - x_{t-p}||_2 < eps for 2 <= p <= max_period.
    Returns (cycle_start_index, period) or (None, None) if not found.
    """
    T = traj.size(0)
    for p in range(2, max_period + 1):
        for t in range(p, T):
            if torch.norm(traj[t] - traj[t - p]) < eps:
                # Optionally backtrack one period to mark earliest cycle index
                return t - p, p
    return None, None


# Random trials
def test_random_points(model, trials=1000, iters=1000, tol=1e-4):
    """
    Randomly initializes x in [0,1], repeatedly applies model(x),
    and checks if the model converges to a fixed point.
    """
    fixed_point_count = 0
    cycles_detected = 0

    for _ in range(trials):
        # start at a random point in the MNIST input space
        x = torch.rand((1, 1, 28, 28)).to(device)

        # Maps trajectories of given trial
        trajectory = []

        for i in range(iters):
            trajectory.append(x.detach().clone())
            x_next = model(x)
            x = x_next.detach()
            
        trajectory_tensor = torch.stack([t.flatten() for t in trajectory])

        # If it detects a fixed point, increment the fixed point count
        fp_idx = detect_fixed_point(trajectory_tensor)
        cyc_start, cyc_period = detect_cycle(trajectory_tensor)

        if fp_idx is not None:
            fixed_point_count += 1

        if cyc_start is not None:
            cycles_detected += 1

    
    return fixed_point_count, cycles_detected

overall_results = {}   # <-- add this BEFORE the for-loop

print("\n===== RANDOM DYNAMICAL TEST RESULTS =====\n")

for name, model in models.items():
    model.eval()
    fp_hits, cycle_hits = test_random_points(model, trials=1000, iters=1000)

    overall_results[name] = {
        "fixed": fp_hits,
        "cycles": cycle_hits,
        "neither": 1000 - (fp_hits + cycle_hits)
    }

    print(f"{name}: fixed={fp_hits}/1000, cycles={cycle_hits}/1000")
import matplotlib.pyplot as plt

# ----- BAR CHART (fixed vs cycles vs neither) -----
labels = list(overall_results.keys())
fixed_vals  = [overall_results[m]["fixed"] for m in labels]
cycle_vals  = [overall_results[m]["cycles"] for m in labels]
neither_vals = [overall_results[m]["neither"] for m in labels]

plt.figure(figsize=(10,6))
plt.bar(labels, fixed_vals, label="Fixed Points")
plt.bar(labels, cycle_vals, bottom=fixed_vals, label="Cycles")
plt.bar(labels, neither_vals, bottom=[fixed_vals[i] + cycle_vals[i] for i in range(len(labels))], label="Neither")
plt.ylabel("Count out of 1000 trials")
plt.title("Dynamical Behavior of Models (Random Trials)")
plt.legend()
plt.show()

# ----- LINE GRAPH (comparison of each behavior type) -----
plt.figure(figsize=(10,6))
plt.plot(labels, fixed_vals, marker='o', label="Fixed Points")
plt.plot(labels, cycle_vals, marker='o', label="Cycles")
plt.plot(labels, neither_vals, marker='o', label="Neither")
plt.ylabel("Count out of 1000 trials")
plt.title("Comparison of behaviors across models")
plt.legend()
plt.show()



