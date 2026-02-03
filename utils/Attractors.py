import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ========= 1) Trajectory generation =========

def iterate_map(
    f: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int = 500,
    device: str = "cpu",
    flatten: bool = True,
) -> torch.Tensor:
    """
    Iteratively apply x_{k+1} = f(x_k), returning the full trajectory (steps+1, D).
    - f: must accept a tensor of shape (1, D) and return shape (1, D).
    - x0: (D,), (1,D), or (1,1,H,W). If flatten=True, images are flattened to (1,D).
    """
    with torch.no_grad():
            x = x0
            if x.dim() == 4 and flatten:      # image -> flatten
                x = x.view(1, -1)
            elif x.dim() == 1:
                x = x.view(1, -1)
            elif x.dim() == 2:
                pass
            else:
                raise ValueError("x0 must be (D,), (1,D), or (1,1,H,W) if flatten=True.")

            x = x.to(device)
            traj = [x.squeeze(0).detach().cpu()]
            for _ in range(steps):
                x = f(x)                # must produce same shape as input (1,D)
                traj.append(x.squeeze(0).detach().cpu())
            traj = torch.stack(traj, dim=0)  # (steps+1, D)
    return traj

# ========= 2) Simple attractor detectors =========

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

@dataclass
class IterationResult:
    traj: torch.Tensor

