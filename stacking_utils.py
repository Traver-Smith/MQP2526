import torch
import torch.nn as nn


def forward_k(model: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Feed 'x' through 'model' k times (re-feeds).
    """
    for _ in range(k):
        x = model(x)
    return x

def stacked_dae_loss(model, noisy, clean, k, weights=None):
    h = noisy
    losses = []
    for t in range(k):
        h = model(h)
        losses.append(torch.mean((h - clean) ** 2))  # MSE_t

    losses = torch.stack(losses)  # [k]
    if weights is None:
        weights = torch.ones(k, device=losses.device) / k
    else:
        weights = weights.to(losses.device)
        weights = weights / weights.sum()

    L_rec = (weights * losses).sum()

    # Optional monotonicity regularizer
    mono = torch.relu(losses[1:] - losses[:-1]).sum()
    L = L_rec + 0.0 * mono

    return L, h   # train objective + final reconstruction