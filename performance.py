import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from .data_processing import add_noise
@torch.no_grad()
def plot_refeed_curves(models, loader, device, max_refeeds=5, criterion=torch.nn.MSELoss()):
    """
    Plot reconstruction loss vs. number of re-feeds for each model.
    Adds reference lines for the 1-pass (baseline) loss.

    Args:
        models (dict): {"AE": model_ae, "DAE": model_dae, "sDAE": model_sdae}
        loader: DataLoader with clean test data
        device: torch.device
        max_refeeds (int): maximum number of times to re-feed the model
        criterion: loss function (default: MSE)
    """
    plt.figure(figsize=(10, 6))

    for name, model in models.items():
        model.eval()
        refeed_losses = []

        for k in range(1, max_refeeds + 1):
            total, count = 0.0, 0
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)

                # Choose input (noisy for DAE/sDAE, clean for AE)
                if name == "AE":
                    inputs = imgs
                else:
                    inputs = imgs + 0.81 * torch.randn_like(imgs)
                    inputs = torch.clamp(inputs, 0., 1.)

                out = inputs
                for _ in range(k):
                    out = model(out)

                loss = criterion(out, imgs)
                bsz = imgs.size(0)
                total += loss.item() * bsz
                count += bsz

            refeed_losses.append(total / max(count, 1))

        # Plot loss vs. re-feeds and capture line handle
        (line,) = plt.plot(range(1, max_refeeds + 1), refeed_losses,
                           marker="o", label=f"{name} Loss")

        # Use same color for baseline
        plt.axhline(y=refeed_losses[0], linestyle="--", alpha=0.7,
                    color=line.get_color(), label=f"{name} 1-pass reference")

    plt.xlabel("Number of Re-feeds (k)")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Reconstruction Loss vs. Number of Re-feeds")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss_curves(history, num_epochs, show_sdae_multi=False):
    """
    Plot training and validation losses for AE, DAE, and sDAE.
    By default, all curves are final-pass MSE for comparability.
    Optionally overlay sDAE's multi-step training loss as a diagnostic.

    Args:
        history (dict): loss history dictionary with keys:
            - "AE": {"train": [], "val": []}
            - "DAE": {"train": [], "val": []}
            - "sDAE": {"train": [], "val_final": [], "val_multi": [], "train_multi": [] (optional)}
        num_epochs (int): number of epochs
        show_sdae_multi (bool): whether to overlay sDAE's multi-step train loss
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10, 6))

    # AE
    plt.plot(epochs, history["AE"]["train"], 'b--', label="AE Train")
    plt.plot(epochs, history["AE"]["val"], 'b-', label="AE Val")

    # DAE
    plt.plot(epochs, history["DAE"]["train"], 'g--', label="DAE Train")
    plt.plot(epochs, history["DAE"]["val"], 'g-', label="DAE Val")

    # sDAE (final-pass for comparability)
    plt.plot(epochs, history["sDAE"]["train"], 'r--', label="sDAE Train (final)")
    plt.plot(epochs, history["sDAE"]["val_final"], 'r-', label="sDAE Val (final)")

    # Optional: overlay diagnostic multi-step train loss
    if show_sdae_multi and "train_multi" in history["sDAE"]:
        plt.plot(epochs, history["sDAE"]["train_multi"], 'r:', label="sDAE Train (multi-step)")

    # Optional: overlay diagnostic val_multi if desired
    # plt.plot(epochs, history["sDAE"]["val_multi"], 'r:', label="sDAE Val (multi-step)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Autoencoder Training/Validation Loss (Final-Pass)")
    plt.legend()
    plt.grid(True)
    plt.show()

@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             *,
             denoising: bool = False,
             re_feeds: int = 1,
             stacked: bool = False,
             multi_step: bool = False,
             noise_factor: float = 0.81,
             device: torch.device = torch.device("cpu"),
             criterion: nn.Module = nn.MSELoss()) -> float:
    """
    Evaluate a model on clean MNIST reconstruction.

    Args:
        model: autoencoder
        loader: DataLoader
        denoising: if True, add Gaussian noise to inputs
        re_feeds: number of re-feeds (default 1)
        stacked: if True, compute stacked loss across steps
        multi_step: if True, return multi-step loss; otherwise final-step loss
        noise_factor: noise std for denoising AEs
        device: torch device
        criterion: loss function (default MSE)

    Returns:
        Average loss across dataset
    """
    model.eval()
    total, count = 0.0, 0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        inputs = add_noise(imgs, noise_factor) if denoising else imgs

        if stacked:
            # stacked DAE: unroll k steps
            h = inputs
            losses = []
            for _ in range(re_feeds):
                h = model(h)
                losses.append(criterion(h, imgs))
            if multi_step:
                loss = torch.stack(losses).mean()   # average across steps
            else:
                loss = losses[-1]                   # final step only
        else:
            # AE or single-pass DAE
            recon = model(inputs)
            loss = criterion(recon, imgs)

        bsz = imgs.size(0)
        total += loss.item() * bsz
        count += bsz

    return total / max(count, 1)