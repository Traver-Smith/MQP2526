"""
Multi-seed attractor analysis for the iterated DAE on MNIST.

For each seed:
  1. Train DAE_SimpleSmall (784→256→64→16→64→256→784) on noisy MNIST
  2. Compute epsilon = mean intra-class NN distance in 16-d latent space
  3. Iterate the latent-to-latent map g(z) = encoder(decoder(z)) for 1000 steps
  4. Determine convergence: earliest step where suffix-sum of remaining
     step distances < epsilon/10
  5. Use the converged iterate (at halt index) as each trajectory's endpoint
  6. Cluster z_first and z_final with KMeans, select k as smallest on
     the silhouette plateau (>= 98% of peak)
  7. Show PCA plots, attractor prototypes (medoids), and label distributions
"""

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# ── Output dirs ──────────────────────────────────────────────────────
OUT_DIR = "multi_seed_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────
SEEDS = [0, 1, 2]
LATENT_DIM = 16
NOISE_SIGMA = 0.65
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-3
NUM_STEPS = 1000
EPS_FRACTION = 0.1          # halting tolerance = eps_fraction * epsilon_latent
PLATEAU_THRESHOLD = 0.98     # first k with silhouette >= 98% of peak
K_MIN, K_MAX = 4, 40


# ── Device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


# ── Data ─────────────────────────────────────────────────────────────
class NoisyMNIST(Dataset):
    def __init__(self, mnist_dataset, sigma=0.65):
        self.mnist = mnist_dataset
        self.sigma = sigma

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        y, label = self.mnist[idx]
        noise = torch.randn_like(y) * self.sigma
        x = torch.clamp(y + noise, 0.0, 1.0)
        return x, y, label


transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)


# ── Model ────────────────────────────────────────────────────────────
class DAE_SimpleSmall(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64, 256),         nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Training ─────────────────────────────────────────────────────────
def train_dae(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(
        NoisyMNIST(train_data, sigma=NOISE_SIGMA),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        NoisyMNIST(test_data, sigma=NOISE_SIGMA),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = DAE_SimpleSmall(latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    prev_val = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for x_noisy, x_clean, _ in train_loader:
            x_noisy, x_clean = x_noisy.to(DEVICE), x_clean.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_noisy), x_clean)
            loss.backward()
            optimizer.step()
            running += loss.item() * x_noisy.size(0)

        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x_noisy, x_clean, _ in val_loader:
                x_noisy, x_clean = x_noisy.to(DEVICE), x_clean.to(DEVICE)
                val_running += criterion(model(x_noisy), x_clean).item() * x_noisy.size(0)
        val_loss = val_running / len(val_loader.dataset)

        print(f"  [seed {seed}] Epoch {epoch+1}/{NUM_EPOCHS}  "
              f"Train: {train_loss:.5f}  Val: {val_loss:.5f}")

        if epoch > 1 and abs(val_loss - prev_val) < 1e-5:
            print(f"  [seed {seed}] Validation plateaued — stopping early.")
            break
        prev_val = val_loss

    return model


# ── Epsilon in latent space ──────────────────────────────────────────
@torch.no_grad()
def compute_epsilon_latent(model):
    model.eval()
    class_groups = {k: [] for k in range(10)}
    for img, label in test_data:
        img = img.unsqueeze(0).to(DEVICE)
        z = model.encoder(img).squeeze(0)
        class_groups[label].append(z)

    avg_nn = {}
    for digit in range(10):
        Z = torch.stack(class_groups[digit])
        D = torch.cdist(Z, Z, p=2)
        D.fill_diagonal_(float("inf"))
        avg_nn[digit] = D.min(dim=1).values.mean().item()

    epsilon = sum(avg_nn.values()) / 10.0
    return epsilon, avg_nn


@torch.no_grad()
def encode_clean_test(model):
    """Encode the clean MNIST test set into latent space."""
    model.eval()
    zs = []
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        zs.append(model.encoder(imgs).cpu())
    return torch.cat(zs, dim=0)


# ── Trajectory iteration in latent space ─────────────────────────────
def find_halt(steps, eps):
    """Earliest index i where sum_{t=i}^{T-1} steps[t] <= eps."""
    hlt = len(steps)
    suffix = 0.0
    for i in range(len(steps) - 1, -1, -1):
        suffix += steps[i]
        if suffix <= eps:
            hlt = i
        else:
            break
    return hlt


@torch.no_grad()
def iterate_trajectories(model, num_steps, eps_halt):
    """
    Iterate the latent-to-latent map g(z) = encoder(decoder(z)) for
    each test image.

    Returns z_first, z_final, halt_indices, step_dists.
    z_final is always the state after all num_steps iterations.
    """
    model.eval()
    encoder, decoder = model.encoder, model.decoder

    loader = DataLoader(
        NoisyMNIST(test_data, sigma=NOISE_SIGMA),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    z_first_list = []
    z_final_list = []
    halt_list = []
    step_dists_list = []
    label_list = []

    for x_noisy, _, labels in tqdm(loader, desc="  Iterating trajectories"):
        x_noisy = x_noisy.to(DEVICE)
        B = x_noisy.size(0)

        z_t = encoder(x_noisy)          # [B, 16]
        batch_dists = []

        for t in range(num_steps):
            z_next = encoder(decoder(z_t))
            d = torch.norm(z_next - z_t, p=2, dim=1)
            batch_dists.append(d.unsqueeze(1))

            if t == 0:
                z_first_list.append(z_next.cpu())

            z_t = z_next

        z_final_list.append(z_t.cpu())
        batch_dists = torch.cat(batch_dists, dim=1)  # [B, num_steps]
        step_dists_list.append(batch_dists.cpu())

        for b in range(B):
            d_row = batch_dists[b].tolist()
            halt_list.append(find_halt(d_row, eps_halt))

        label_list.append(labels)

    z_first = torch.cat(z_first_list, dim=0)
    z_final = torch.cat(z_final_list, dim=0)
    halt_indices = np.array(halt_list, dtype=np.int64)
    step_dists = torch.cat(step_dists_list, dim=0).numpy()
    all_labels = torch.cat(label_list, dim=0).numpy()

    return z_first, z_final, halt_indices, step_dists, all_labels


# ── Clustering ───────────────────────────────────────────────────────
def select_k_and_cluster(z_np, k_min=K_MIN, k_max=K_MAX):
    """KMeans sweep; select smallest k on the silhouette plateau."""
    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(z_np)
        distinct = int(np.unique(labels).size)
        sil = float(silhouette_score(z_np, labels)) if distinct > 1 else float("nan")
        results.append({"k": k, "kmeans": km, "labels": labels,
                         "silhouette": sil, "distinct": distinct})

    valid = [r for r in results if r["distinct"] == r["k"]]
    if not valid:
        valid = results

    peak_sil = max(r["silhouette"] for r in valid)
    floor = PLATEAU_THRESHOLD * peak_sil
    on_plateau = [r for r in valid if r["silhouette"] >= floor]
    selected = min(on_plateau, key=lambda r: r["k"])

    return selected, results


# ── Medoid prototypes ────────────────────────────────────────────────
def get_medoids(z_np, labels, centers_np, best_k):
    """For each cluster, find the actual data point closest to centroid."""
    medoid_indices = np.zeros(best_k, dtype=int)
    medoid_latents = np.zeros_like(centers_np)
    for c in range(best_k):
        mask = labels == c
        pts = z_np[mask]
        global_idx = np.where(mask)[0]
        dists = np.linalg.norm(pts - centers_np[c], axis=1)
        best = int(dists.argmin())
        medoid_indices[c] = global_idx[best]
        medoid_latents[c] = pts[best]
    return medoid_indices, medoid_latents


# ── Plotting helpers (poster-sized fonts) ────────────────────────────
TITLE_SIZE = 36
LABEL_SIZE = 30
TICK_SIZE = 24
LEGEND_SIZE = 22
ANNOT_SIZE = 22


def plot_convergence(step_dists, halt_indices, epsilon_latent, eps_halt, seed):
    N, T = step_dists.shape
    steps = np.arange(1, T + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    median_d = np.median(step_dists, axis=0)
    p25 = np.percentile(step_dists, 25, axis=0)
    p75 = np.percentile(step_dists, 75, axis=0)
    p10 = np.percentile(step_dists, 10, axis=0)
    p90 = np.percentile(step_dists, 90, axis=0)

    ax1.semilogy(steps, median_d, color="tab:blue", lw=3, label="Median")
    ax1.fill_between(steps, p25, p75, alpha=0.3, color="tab:blue", label="25th-75th pctl")
    ax1.fill_between(steps, p10, p90, alpha=0.12, color="tab:blue", label="10th-90th pctl")
    ax1.axhline(eps_halt, color="tab:red", ls="--", lw=2.5,
                label=f"Halting eps = {eps_halt:.4f}")
    ax1.axhline(epsilon_latent, color="tab:orange", ls=":", lw=2.5,
                label=f"eps_latent = {epsilon_latent:.3f}")
    ax1.set_xlabel("Iteration t", fontsize=LABEL_SIZE)
    ax1.set_ylabel("||g(z_t) - z_t||  (log scale)", fontsize=LABEL_SIZE)
    ax1.set_title("Per-step latent distance", fontsize=TITLE_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, alpha=0.3)

    frac = np.array([(halt_indices <= t).sum() / N for t in range(T)])
    ax2.plot(steps, frac * 100, color="tab:green", lw=3)
    ax2.axhline(100, color="gray", ls=":", alpha=0.5)
    for pct, c, ls in [(50, "tab:orange", "--"), (90, "tab:red", "--"), (99, "tab:purple", ":")]:
        idx = np.searchsorted(frac, pct / 100.0)
        if idx < T:
            ax2.axvline(idx + 1, color=c, ls=ls, alpha=0.7, lw=2,
                        label=f"{pct}% at step {idx+1}")
            ax2.plot(idx + 1, pct, "o", color=c, ms=10)
    ax2.set_xlabel("Iteration t", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Cumulative % converged", fontsize=LABEL_SIZE)
    ax2.set_title("Convergence rate", fontsize=TITLE_SIZE)
    ax2.legend(fontsize=LEGEND_SIZE, loc="lower right")
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.set_ylim(-2, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"seed{seed}_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_silhouette(results, selected_k, seed, tag):
    ks = [r["k"] for r in results]
    sils = [r["silhouette"] for r in results]
    peak_k = max(results, key=lambda r: r["silhouette"])["k"]

    plt.figure(figsize=(10, 7))
    plt.plot(ks, sils, marker="o", ms=8, lw=2.5)
    plt.axvline(selected_k, color="tab:red", ls="-", lw=2.5,
                label=f"Selected k={selected_k} (first on plateau)")
    if peak_k != selected_k:
        plt.axvline(peak_k, color="tab:gray", ls="--", lw=2,
                    label=f"Peak k={peak_k}")
    plt.xlabel("Number of clusters (k)", fontsize=LABEL_SIZE)
    plt.ylabel("Silhouette score", fontsize=LABEL_SIZE)
    plt.title(f"Seed {seed}: Silhouette ({tag})", fontsize=TITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tick_params(labelsize=TICK_SIZE)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"seed{seed}_silhouette_{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_pca(z_np, labels, medoid_latents, proto_imgs, best_k, seed, tag, title=None):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    pca = PCA(n_components=2, random_state=42)
    z_2d = pca.fit_transform(z_np)
    c_2d = pca.transform(medoid_latents)

    fig, ax = plt.subplots(figsize=(16, 14))

    # Scatter all points
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="tab20",
               s=36, alpha=0.85, linewidths=0)

    # Small colored dots at cluster centers (no big numbered circles)
    ax.scatter(c_2d[:, 0], c_2d[:, 1], s=120, c=np.arange(best_k),
               cmap="tab20", edgecolors="black", linewidths=1.5, zorder=4)

    ax.set_xlabel("PC 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("PC 2", fontsize=LABEL_SIZE)
    ax.set_title(title or f"PCA of {tag} (k={best_k})", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.margins(0.25)  # extra margin so annotations have room

    # Annotate the largest clusters with decoded centroid images
    MAX_ANNOTATIONS = 10
    if proto_imgs is not None:
        # Pick the top clusters by size
        cluster_sizes = np.bincount(labels, minlength=best_k)
        top_cids = np.argsort(cluster_sizes)[::-1][:MAX_ANNOTATIONS]

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_range = x_max - x_min
        y_range = y_max - y_min

        cx_mean = c_2d[:, 0].mean()
        cy_mean = c_2d[:, 1].mean()

        # Sort the selected clusters by angle so images don't cross
        angles = np.arctan2(c_2d[top_cids, 1] - cy_mean,
                            c_2d[top_cids, 0] - cx_mean)
        top_cids = top_cids[np.argsort(angles)]

        for cid in top_cids:
            angle = np.arctan2(c_2d[cid, 1] - cy_mean,
                               c_2d[cid, 0] - cx_mean)
            ax_x = cx_mean + 0.85 * (x_range / 2) * np.cos(angle)
            ax_y = cy_mean + 0.85 * (y_range / 2) * np.sin(angle)

            img = proto_imgs[cid, 0]
            imagebox = OffsetImage(img, zoom=1.8, cmap="gray")
            imagebox.image.axes = ax

            ab = AnnotationBbox(
                imagebox, (c_2d[cid, 0], c_2d[cid, 1]),
                xybox=(ax_x, ax_y),
                xycoords="data", boxcoords="data",
                arrowprops=dict(arrowstyle="-", color="black", lw=1.5),
                bboxprops=dict(edgecolor="black", lw=1.5, facecolor="white"),
                pad=0.3,
                zorder=6,
            )
            ax.add_artist(ab)

            ax.annotate(
                f"C{cid}", (ax_x, ax_y),
                fontsize=ANNOT_SIZE - 2, weight="bold",
                ha="center", va="bottom",
                xytext=(0, 22), textcoords="offset points",
                zorder=7,
            )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"seed{seed}_pca_{tag}_k{best_k}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_attractor_profiles(proto_imgs, labels_final, true_labels,
                            cluster_sizes, cluster_label_counts,
                            cluster_label_props, best_k, seed):
    """
    One figure per cluster combining:
      Row 1: decoded centroid prototype (large) | exemplar clean MNIST images
             from each significant digit class (>= 1% of cluster)
      Row 2: bar chart of full label distribution spanning the figure
    """
    cdir = os.path.join(OUT_DIR, f"seed{seed}_attractor_profiles")
    os.makedirs(cdir, exist_ok=True)

    cluster_mask_all = {}
    cluster_indices_all = {}
    cluster_true_all = {}
    for cid in range(best_k):
        mask = labels_final == cid
        cluster_mask_all[cid] = mask
        cluster_indices_all[cid] = np.where(mask)[0]
        cluster_true_all[cid] = true_labels[mask]

    for cid in range(best_k):
        counts = cluster_label_counts[cid]
        total = int(counts.sum())
        if total == 0:
            continue
        present_digits = np.where(counts / total >= 0.01)[0]
        n_examples = len(present_digits)

        # Layout: top row has prototype + exemplars, bottom row is bar chart
        n_img_cols = 1 + n_examples
        fig = plt.figure(figsize=(max(3.5 * n_img_cols, 8), 9))
        gs = fig.add_gridspec(2, n_img_cols, height_ratios=[3, 1.2],
                              hspace=0.35, wspace=0.25)

        # ── Top-left: prototype ──
        ax_proto = fig.add_subplot(gs[0, 0])
        ax_proto.imshow(proto_imgs[cid, 0], cmap="gray")
        ax_proto.set_xticks([])
        ax_proto.set_yticks([])
        ax_proto.set_title("Decoded\ncentroid", fontsize=ANNOT_SIZE)
        # Border to distinguish from exemplars
        for spine in ax_proto.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)

        # ── Top row, cols 1+: exemplar clean MNIST images ──
        cluster_indices = cluster_indices_all[cid]
        cluster_true = cluster_true_all[cid]

        for col, digit in enumerate(present_digits, start=1):
            ax = fig.add_subplot(gs[0, col])
            digit_in_cluster = cluster_indices[cluster_true == digit]
            rng = np.random.RandomState(seed * 1000 + cid * 10 + digit)
            sample_idx = rng.choice(digit_in_cluster)
            clean_img, _ = test_data[sample_idx]

            ax.imshow(clean_img[0].numpy(), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            n_this = int(counts[digit])
            pct = 100 * n_this / total
            ax.set_title(f"Label {digit}\nn={n_this} ({pct:.0f}%)",
                         fontsize=ANNOT_SIZE - 2)

        # ── Bottom row: label distribution bar chart spanning full width ──
        ax_bar = fig.add_subplot(gs[1, :])
        props = cluster_label_props[cid]
        colors = plt.cm.tab10(np.arange(10))
        bars = ax_bar.bar(range(10), props, color=colors, width=0.8)
        ax_bar.set_xticks(range(10))
        ax_bar.set_xticklabels([str(d) for d in range(10)], fontsize=TICK_SIZE)
        ax_bar.set_xlabel("True MNIST digit", fontsize=LABEL_SIZE - 4)
        ax_bar.set_ylabel("Fraction", fontsize=LABEL_SIZE - 4)
        ax_bar.set_ylim(0, min(1.0, props.max() * 1.3))
        ax_bar.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax_bar.grid(axis="y", alpha=0.3)

        # Percentage labels on bars
        for d in range(10):
            if props[d] >= 0.01:
                ax_bar.text(d, props[d] + 0.01, f"{props[d]*100:.0f}%",
                            ha="center", fontsize=ANNOT_SIZE - 6, weight="bold")

        n = cluster_sizes[cid]
        top = int(props.argmax())
        fig.suptitle(f"Cluster {cid}   (n={n}, dominant label: {top})",
                     fontsize=TITLE_SIZE - 4, y=1.02)
        fig.tight_layout()
        path = os.path.join(cdir, f"cluster_{cid}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"  Saved attractor profiles -> {cdir}/")


def plot_silhouette_comparison(res_first, sel_first, res_final, sel_final, seed):
    """Overlay silhouette curves for pre- and post-iteration on one plot."""
    ks_first = [r["k"] for r in res_first]
    sils_first = [r["silhouette"] for r in res_first]
    ks_final = [r["k"] for r in res_final]
    sils_final = [r["silhouette"] for r in res_final]

    plt.figure(figsize=(12, 9))
    plt.plot(ks_first, sils_first, marker="o", ms=8, lw=2.5,
             color="tab:blue", label="Before iteration (z_first)")
    plt.plot(ks_final, sils_final, marker="s", ms=8, lw=2.5,
             color="tab:orange", label="After iteration (z_final)")
    plt.xlabel("Number of clusters (k)", fontsize=LABEL_SIZE)
    plt.ylabel("Silhouette score", fontsize=LABEL_SIZE)
    plt.title("Silhouette: before vs after iteration", fontsize=TITLE_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tick_params(labelsize=TICK_SIZE)
    plt.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"seed{seed}_silhouette_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def label_summary(labels, orig_labels, best_k, num_classes=10):
    sizes = np.zeros(best_k, dtype=int)
    counts = np.zeros((best_k, num_classes), dtype=int)
    props = np.zeros((best_k, num_classes), dtype=float)
    for c in range(best_k):
        mask = labels == c
        sizes[c] = int(mask.sum())
        members = orig_labels[mask]
        if members.size == 0:
            continue
        bc = np.bincount(members, minlength=num_classes)
        counts[c] = bc
        props[c] = bc / bc.sum()
    return sizes, counts, props


# ── Per-seed pipeline ────────────────────────────────────────────────
def run_seed(seed):
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # 1. Train
    model = train_dae(seed)

    # 2. Epsilon
    epsilon_latent, avg_nn = compute_epsilon_latent(model)
    eps_halt = EPS_FRACTION * epsilon_latent
    print(f"  epsilon_latent = {epsilon_latent:.6f}")
    print(f"  halting eps    = {eps_halt:.6f}  (10% of epsilon_latent)")

    # 2b. Encode clean MNIST test set, cluster, and plot PCA
    z_clean = encode_clean_test(model)
    z_clean_np = z_clean.numpy()
    sel_clean, res_clean = select_k_and_cluster(z_clean_np)
    k_clean = sel_clean["k"]
    labels_clean = sel_clean["labels"]
    centers_clean = sel_clean["kmeans"].cluster_centers_
    print(f"  z_clean: selected k={k_clean} (sil={sel_clean['silhouette']:.4f})")

    plot_silhouette(res_clean, k_clean, seed, "z_clean")

    # Decode centroids for annotation
    model.decoder.eval()
    with torch.no_grad():
        clean_proto = model.decoder(
            torch.from_numpy(centers_clean).float().to(DEVICE)
        ).cpu().numpy()
    plot_pca(z_clean_np, labels_clean, centers_clean, None, k_clean, seed, "z_clean",
             title=f"Clean MNIST embeddings (k={k_clean})")

    # 3. Iterate trajectories (latent-to-latent, 1000 steps)
    z_first, z_final, halt_indices, step_dists, true_labels = \
        iterate_trajectories(model, NUM_STEPS, eps_halt)

    N = z_first.shape[0]
    never = (halt_indices >= NUM_STEPS).sum()
    print(f"  Trajectories: {N}")
    print(f"  Median halt step: {int(np.median(halt_indices))}")
    print(f"  99th pctl halt:   {int(np.percentile(halt_indices, 99))}")
    print(f"  Never converged:  {never} ({100*never/N:.2f}%)")

    # 4. Convergence plot
    plot_convergence(step_dists, halt_indices, epsilon_latent, eps_halt, seed)

    # 5. Cluster z_first
    z_first_np = z_first.numpy()
    sel_first, res_first = select_k_and_cluster(z_first_np)
    k_first = sel_first["k"]
    labels_first = sel_first["labels"]
    centers_first = sel_first["kmeans"].cluster_centers_
    print(f"  z_first: selected k={k_first} (sil={sel_first['silhouette']:.4f})")

    plot_silhouette(res_first, k_first, seed, "z_first")

    # Medoids for z_first (no decoder — just PCA)
    med_idx_f, med_lat_f = get_medoids(z_first_np, labels_first, centers_first, k_first)
    plot_pca(z_first_np, labels_first, med_lat_f, None, k_first, seed, "z_first",
             title=f"After one DAE pass, noisy input (k={k_first})")

    # 6. Cluster z_converged
    z_conv_np = z_final.numpy()
    sel_final, res_final = select_k_and_cluster(z_conv_np)
    k_final = sel_final["k"]
    labels_final = sel_final["labels"]
    centers_final = sel_final["kmeans"].cluster_centers_
    print(f"  z_converged: selected k={k_final} (sil={sel_final['silhouette']:.4f})")

    plot_silhouette(res_final, k_final, seed, "z_converged")

    # Combined silhouette comparison
    plot_silhouette_comparison(res_first, sel_first, res_final, sel_final, seed)

    # Decoded centroid prototypes
    model.decoder.eval()
    with torch.no_grad():
        proto_imgs = model.decoder(
            torch.from_numpy(centers_final).float().to(DEVICE)
        ).cpu().numpy()

    plot_pca(z_conv_np, labels_final, centers_final, proto_imgs, k_final, seed, "z_converged",
             title=f"After 1000 iterations (k={k_final})")

    # 7. Combined attractor profiles: prototype + exemplars + label distribution
    sizes, counts, props = label_summary(labels_final, true_labels, k_final)
    plot_attractor_profiles(proto_imgs, labels_final, true_labels,
                            sizes, counts, props, k_final, seed)

    print(f"\n  Attractor composition (seed {seed}, k={k_final}):")
    for c in range(k_final):
        top = int(props[c].argmax())
        top_p = props[c][top] * 100
        print(f"    C{c}: n={sizes[c]:5d}, top label={top} ({top_p:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for seed in SEEDS:
        run_seed(seed)
    print(f"\nAll done. Results in ./{OUT_DIR}/")
