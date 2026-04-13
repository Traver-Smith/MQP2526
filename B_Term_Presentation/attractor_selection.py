import math
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _to_numpy_labels(orig_labels: Optional[Any]) -> Optional[np.ndarray]:
    if orig_labels is None:
        return None
    if isinstance(orig_labels, torch.Tensor):
        return orig_labels.cpu().numpy()
    return np.asarray(orig_labels)


def _weighted_purity(labels: np.ndarray, orig_labels: np.ndarray, num_classes: int) -> float:
    purity_total = 0
    for cluster_id in np.unique(labels):
        members = orig_labels[labels == cluster_id]
        if members.size == 0:
            continue
        counts = np.bincount(members, minlength=num_classes)
        purity_total += int(counts.max())
    return purity_total / len(orig_labels)


def _label_summary(
    labels: np.ndarray,
    orig_labels: np.ndarray,
    best_k: int,
    num_classes: int,
) -> Dict[str, np.ndarray]:
    cluster_sizes = np.zeros(best_k, dtype=int)
    cluster_label_counts = np.zeros((best_k, num_classes), dtype=int)
    cluster_label_props = np.zeros((best_k, num_classes), dtype=float)

    for cluster_id in range(best_k):
        mask = labels == cluster_id
        cluster_sizes[cluster_id] = int(mask.sum())
        cluster_members = orig_labels[mask]
        if cluster_members.size == 0:
            continue
        counts = np.bincount(cluster_members, minlength=num_classes)
        cluster_label_counts[cluster_id] = counts
        cluster_label_props[cluster_id] = counts / counts.sum()

    return {
        "cluster_sizes": cluster_sizes,
        "cluster_label_counts": cluster_label_counts,
        "cluster_label_props": cluster_label_props,
    }


def _evaluate_k_grid(
    z_np: np.ndarray,
    k_values: List[int],
    orig_labels_np: Optional[np.ndarray],
    num_classes: int,
    random_state: int,
    n_init: int,
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = kmeans.fit_predict(z_np)
        distinct_clusters = int(np.unique(labels).size)
        sizes = np.bincount(labels, minlength=k)
        silhouette = float(silhouette_score(z_np, labels)) if distinct_clusters > 1 else float("nan")

        ari = None
        nmi = None
        purity = None
        if orig_labels_np is not None:
            ari = float(adjusted_rand_score(orig_labels_np, labels))
            nmi = float(normalized_mutual_info_score(orig_labels_np, labels))
            purity = float(_weighted_purity(labels, orig_labels_np, num_classes))

        metrics.append(
            {
                "k": k,
                "kmeans": kmeans,
                "labels": labels,
                "silhouette": silhouette,
                "distinct_clusters": distinct_clusters,
                "cluster_sizes": sizes,
                "ari": ari,
                "nmi": nmi,
                "purity": purity,
            }
        )

    return metrics


def _select_best_metric(
    metrics: List[Dict[str, Any]],
    plateau_threshold: float = 0.98,
) -> Dict[str, Any]:
    """
    Pick the smallest k whose silhouette score reaches within
    (1 - plateau_threshold) of the peak.  This is the first k on the
    plateau — the most parsimonious clustering that captures the
    attractor structure.
    """
    valid_metrics = [m for m in metrics if m["distinct_clusters"] == m["k"]]
    if not valid_metrics:
        valid_metrics = metrics

    peak_sil = max(m["silhouette"] for m in valid_metrics)
    floor = plateau_threshold * peak_sil

    # Smallest k that reaches the plateau
    on_plateau = [m for m in valid_metrics if m["silhouette"] >= floor]
    return min(on_plateau, key=lambda m: m["k"])


def _plot_silhouette(
    metrics: List[Dict[str, Any]],
    name: str,
    selected_k: int,
    peak_k: int,
) -> str:
    k_values = [m["k"] for m in metrics]
    silhouette_values = [m["silhouette"] for m in metrics]

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(k_values, silhouette_values, marker="o", label="Silhouette")
    plt.axvline(selected_k, color="tab:red", linestyle="-", label=f"Selected k = {selected_k} (first on plateau)")
    if peak_k != selected_k:
        plt.axvline(peak_k, color="tab:gray", linestyle="--", label=f"Peak silhouette k = {peak_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title(f"Silhouette score vs k for {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    sil_path = f"Photos/{name}_silhouette.png"
    plt.savefig(sil_path, dpi=300, bbox_inches="tight")
    plt.show()
    return sil_path


def _plot_k_diagnostics(metrics: List[Dict[str, Any]], name: str, selected_k: int) -> str:
    k_values = [m["k"] for m in metrics]
    min_cluster_fracs = [int(m["cluster_sizes"].min()) / int(np.sum(m["cluster_sizes"])) for m in metrics]

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(k_values, [m["silhouette"] for m in metrics], marker="o", label="Silhouette", color="tab:blue")

    if metrics[0]["ari"] is not None:
        plt.plot(k_values, [m["ari"] for m in metrics], marker="s", label="ARI vs labels", color="tab:green")
        plt.plot(k_values, [m["nmi"] for m in metrics], marker="^", label="NMI vs labels", color="tab:orange")
        plt.plot(k_values, [m["purity"] for m in metrics], marker="d", label="Weighted purity", color="tab:purple")

    plt.plot(
        k_values,
        min_cluster_fracs,
        marker="x",
        linestyle="--",
        label="Smallest basin fraction",
        color="tab:red",
    )
    plt.axvline(selected_k, color="black", linestyle=":", label=f"Selected k = {selected_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Score")
    plt.title(f"k-selection diagnostics for {name}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    diag_path = f"Photos/{name}_k_diagnostics.png"
    plt.savefig(diag_path, dpi=300, bbox_inches="tight")
    plt.show()
    return diag_path


def plot_silhouette_comparison(
    before_result: Dict[str, Any],
    after_result: Dict[str, Any],
    before_label: str = "Before iteration (z_first)",
    after_label: str = "After iteration (z_final)",
    out_path: str = "Photos/z_first_vs_z_final_silhouette.png",
) -> str:
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(
        before_result["k_values"],
        before_result["sil_scores"],
        marker="o",
        linewidth=2,
        label=before_label,
    )
    plt.plot(
        after_result["k_values"],
        after_result["sil_scores"],
        marker="s",
        linewidth=2,
        label=after_label,
    )
    plt.axvline(
        before_result["best_k"],
        color="tab:blue",
        linestyle=":",
        alpha=0.7,
        label=f"{before_label} selected k = {before_result['best_k']}",
    )
    plt.axvline(
        after_result["best_k"],
        color="tab:orange",
        linestyle=":",
        alpha=0.7,
        label=f"{after_label} selected k = {after_result['best_k']}",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Comparison Before vs After Iteration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined silhouette comparison -> {out_path}")
    return out_path


def _plot_pca_clusters(
    z_2d: np.ndarray,
    labels: np.ndarray,
    centers_2d: np.ndarray,
    name: str,
    best_k: int,
    proto_imgs: Optional[np.ndarray],
) -> str:
    point_size = 28
    point_alpha = 0.95

    if proto_imgs is not None:
        fig = plt.figure(figsize=(12.8, 7.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.8, 1.35], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        side_ax = fig.add_subplot(gs[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        side_ax = None

    scatter = ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=labels,
        cmap="tab20",
        s=point_size,
        alpha=point_alpha,
        linewidths=0.0,
    )

    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        s=320,
        c=np.arange(best_k),
        cmap="tab20",
        edgecolors="black",
        linewidths=1.8,
        zorder=4,
    )

    for cluster_id, (x_coord, y_coord) in enumerate(centers_2d):
        ax.text(
            x_coord,
            y_coord,
            str(cluster_id),
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
            zorder=5,
        )

    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"PCA projection of {name} (k={best_k})")
    ax.margins(0.18)

    if side_ax is not None and proto_imgs is not None:
        side_ax.axis("off")
        side_ax.set_title("Decoded centroids", fontsize=12, pad=10)

        y_positions = np.linspace(0.92, 0.08, best_k)
        for cluster_id, y_pos in enumerate(y_positions):
            side_ax.text(
                0.02,
                y_pos,
                f"C{cluster_id}",
                transform=side_ax.transAxes,
                fontsize=11,
                weight="bold",
                va="center",
                ha="left",
            )
            side_ax.imshow(
                proto_imgs[cluster_id, 0],
                cmap="gray",
                extent=(0.22, 0.92, y_pos - 0.045, y_pos + 0.045),
                aspect="auto",
                zorder=2,
            )
            side_ax.add_patch(
                plt.Rectangle(
                    (0.22, y_pos - 0.045),
                    0.70,
                    0.09,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                    transform=side_ax.transAxes,
                )
            )
        side_ax.set_xlim(0, 1)
        side_ax.set_ylim(0, 1)
    elif side_ax is None:
        fig.colorbar(scatter, ax=ax, label="Cluster ID")

    fig.tight_layout()

    pca_path = f"Photos/{name}_pca_k{best_k}.png"
    fig.savefig(pca_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved PCA cluster plot -> {pca_path}")
    return pca_path


def validate_centroids_as_fixed_points(
    centers_np: np.ndarray,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: str,
    num_iters: int = 200,
) -> Dict[str, Any]:
    """
    Iterate each centroid through the latent-to-latent map g(z) = encoder(decoder(z))
    num_iters more times. Report the per-step residual ||g(z) - z|| in the 16-d
    latent space to verify practical fixed-point status.
    """
    encoder.eval()
    decoder.eval()
    z = torch.from_numpy(centers_np).float().to(device)
    k = z.shape[0]

    with torch.no_grad():
        # Iterate the latent-to-latent map: z -> decoder -> encoder -> z
        for _ in range(num_iters):
            z = encoder(decoder(z))

        # One more step to measure residual in latent space
        z_next = encoder(decoder(z))
        residuals = torch.norm(z_next - z, p=2, dim=1).cpu().numpy()

        z_iterated = z.cpu().numpy()
        # Decode for semantic interpretation
        decoded = decoder(z).cpu().numpy()

    return {
        "residuals": residuals,
        "z_iterated": z_iterated,
        "decoded_fixed_points": decoded,
    }


def cluster_and_plot_latents(
    z_tensor: torch.Tensor,
    name: str = "z",
    k_min: int = 4,
    k_max: int = 40,
    decoder: Optional[torch.nn.Module] = None,
    encoder: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    orig_labels: Optional[Any] = None,
    num_classes: int = 10,
    random_state: int = 42,
    n_init: int = 20,
) -> Dict[str, Any]:
    z_np = z_tensor.detach().cpu().numpy()
    orig_labels_np = _to_numpy_labels(orig_labels)

    if orig_labels_np is not None and orig_labels_np.shape[0] != z_np.shape[0]:
        raise ValueError("orig_labels must have same length as z_tensor")

    print(f"{name} shape: {z_np.shape}")

    k_values = list(range(k_min, k_max + 1))
    metrics = _evaluate_k_grid(
        z_np=z_np,
        k_values=k_values,
        orig_labels_np=orig_labels_np,
        num_classes=num_classes,
        random_state=random_state,
        n_init=n_init,
    )

    for metric in metrics:
        label_msg = ""
        if metric["ari"] is not None:
            label_msg = (
                f", ARI={metric['ari']:.4f}, NMI={metric['nmi']:.4f}, purity={metric['purity']:.4f}"
            )
        print(
            f"[{name}] k={metric['k']:2d}, silhouette={metric['silhouette']:.4f}"
            f"{label_msg}"
        )

    selected = _select_best_metric(metrics)
    silhouette_peak = max(metrics, key=lambda m: m["silhouette"])

    sil_path = _plot_silhouette(
        metrics=metrics,
        name=name,
        selected_k=selected["k"],
        peak_k=silhouette_peak["k"],
    )
    diag_path = _plot_k_diagnostics(metrics=metrics, name=name, selected_k=selected["k"])

    print(
        f"\nSelected k for {name} = {selected['k']} "
        f"(first k on plateau, silhouette = {selected['silhouette']:.4f}, "
        f"peak was k={silhouette_peak['k']} at {silhouette_peak['silhouette']:.4f})"
    )

    labels = selected["labels"]
    best_k = selected["k"]
    best_kmeans = selected["kmeans"]
    centers_np = best_kmeans.cluster_centers_

    # --- Fixed-point validation ---
    fp_validation = None
    if encoder is not None and decoder is not None:
        fp_validation = validate_centroids_as_fixed_points(
            centers_np=centers_np,
            encoder=encoder,
            decoder=decoder,
            device=device,
        )
        print(f"\nFixed-point validation ({name}, k={best_k}):")
        for c in range(best_k):
            r = fp_validation["residuals"][c]
            print(f"  Centroid {c}: residual ||f(x*)-x*|| = {r:.2e}")
        print(f"  Max residual: {fp_validation['residuals'].max():.2e}")
        print(f"  Median residual: {np.median(fp_validation['residuals']):.2e}")

    # --- PCA + plots ---
    pca = PCA(n_components=2, random_state=random_state)
    z_2d = pca.fit_transform(z_np)

    prototypes_path = None
    proto_imgs = None
    if decoder is not None:
        decoder.eval()

        # Use iterated fixed points as prototypes when available —
        # these are the actual attractors, not geometric cluster means
        if fp_validation is not None:
            proto_imgs = fp_validation["decoded_fixed_points"]
            proto_source = "iterated fixed points"
            # Use the iterated latent positions for PCA centroid placement
            centers_for_pca = fp_validation["z_iterated"]
        else:
            centers = torch.from_numpy(centers_np).to(device)
            with torch.no_grad():
                proto_imgs = decoder(centers).cpu().numpy()
            proto_source = "decoded KMeans centroids"
            centers_for_pca = centers_np

        n_cols = min(10, best_k)
        n_rows = math.ceil(best_k / n_cols)
        plt.figure(figsize=(1.5 * n_cols, 1.5 * n_rows))
        for cluster_id in range(best_k):
            ax = plt.subplot(n_rows, n_cols, cluster_id + 1)
            ax.imshow(proto_imgs[cluster_id, 0], cmap="gray")
            ax.set_title(f"{cluster_id}", fontsize=8)
            ax.axis("off")

        plt.suptitle(f"{name} attractor prototypes (k={best_k}, {proto_source})")
        plt.tight_layout()
        prototypes_path = f"Photos/{name}_prototypes_k{best_k}.png"
        plt.savefig(prototypes_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved prototypes for {name} -> {prototypes_path} (source: {proto_source})")

    else:
        centers_for_pca = centers_np

    centers_2d = pca.transform(centers_for_pca)
    pca_path = _plot_pca_clusters(
        z_2d=z_2d,
        labels=labels,
        centers_2d=centers_2d,
        name=name,
        best_k=best_k,
        proto_imgs=proto_imgs,
    )

    cluster_sizes = None
    cluster_label_counts = None
    cluster_label_props = None
    if orig_labels_np is not None:
        summary = _label_summary(
            labels=labels,
            orig_labels=orig_labels_np,
            best_k=best_k,
            num_classes=num_classes,
        )
        cluster_sizes = summary["cluster_sizes"]
        cluster_label_counts = summary["cluster_label_counts"]
        cluster_label_props = summary["cluster_label_props"]

        print(f"\nCluster composition for {name} (k={best_k}):")
        for cluster_id in range(best_k):
            size = cluster_sizes[cluster_id]
            if size == 0:
                print(f"  Cluster {cluster_id}: EMPTY")
                continue
            top_label = int(cluster_label_props[cluster_id].argmax())
            top_prop = cluster_label_props[cluster_id][top_label] * 100.0
            print(f"  Cluster {cluster_id}: n={size}, top label={top_label} ({top_prop:.2f}%)")

    return {
        "k_values": [m["k"] for m in metrics],
        "sil_scores": [m["silhouette"] for m in metrics],
        "metrics": metrics,
        "best_k": best_k,
        "best_score": selected["silhouette"],
        "labels": labels,
        "z_2d": z_2d,
        "prototypes_path": prototypes_path,
        "proto_imgs": proto_imgs,
        "cluster_sizes": cluster_sizes,
        "cluster_label_counts": cluster_label_counts,
        "cluster_label_props": cluster_label_props,
        "silhouette_plot_path": sil_path,
        "k_diagnostics_path": diag_path,
        "fp_validation": fp_validation,
        "selection_metric": {
            "selected_k": best_k,
            "selected_silhouette": selected["silhouette"],
            "selected_ari": selected["ari"],
            "selected_nmi": selected["nmi"],
            "selected_purity": selected["purity"],
        },
    }
