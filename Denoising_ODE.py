# ===================== (A) True Neural ODE: fixed dim + zero-drift =====================
import time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchdiffeq import odeint
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt

# --- Device / dtype ---
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
print("Device:", device)

# --- MNIST (flatten) ---
def flatten(x): return x.view(-1)
transform = T.Compose([T.ToTensor(), T.Lambda(flatten)])
train_full = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_set, _ = random_split(train_full, [55000, len(train_full)-55000])
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=64,  shuffle=False, num_workers=0)

# --- Denoising noise ---
noise_std = 0.81
def add_noise(x): return (x + noise_std*torch.randn_like(x)).clamp(0.,1.)

# --- 6 "layer" intervals with active dimensions (AE widths without changing state dim) ---
widths = [784, 512, 256,  64, 256, 512]   # 6 intervals in [0,6)
K = len(widths)

class SubField(nn.Module):
    """Per-interval velocity with internal bottleneck: 784 -> d_k -> 784, zero drift on inactive coords."""
    def __init__(self, d_active):
        super().__init__()
        self.d = d_active
        self.enc = nn.Sequential(nn.Linear(784, self.d), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(self.d, 784))
    def forward(self, x):
        h  = self.enc(x)
        v  = self.dec(h)                        # full 784-d velocity proposal
        if self.d < 784:
            v[:, self.d:] = 0.0                 # enforce zero drift outside active subspace
        return v

class PiecewiseField(nn.Module):
    """Piecewise f_k over t in [k, k+1): keeps ODE in 784-D; mimics layerwise compression/expansion."""
    def __init__(self, widths):
        super().__init__()
        self.blocks = nn.ModuleList([SubField(d) for d in widths])
    def forward(self, t, x):
        # choose interval k = floor(t) in [0..K-1]
        tau = float(t) if (isinstance(t, float) or t.ndim==0) else float(t.item())
        k = int(max(0, min(len(self.blocks)-1, int(tau // 1))))
        return self.blocks[k](x)

def train_true_ode(epochs=20, lr=1e-3, method='dopri5'):
    func = PiecewiseField(widths).to(device)
    opt  = torch.optim.Adam(func.parameters(), lr=lr)
    print("\n=== (A) Train True Neural ODE (fixed 784-D, zero padding) ===")
    for ep in range(1, epochs+1):
        func.train(); tot, n = 0., 0; t0 = time.time()
        for xb, _ in train_loader:
            xb = xb.to(device); nb = add_noise(xb)
            tspan = torch.tensor([0., 6.], device=device)
            xT = odeint(func, nb, tspan, method=method,
                        rtol=1e-3, atol=1e-4, options={'dtype': torch.float32})[-1]
            loss = F.mse_loss(xT, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0); n += xb.size(0)
        print(f"[A][Ep {ep:02d}] Train MSE: {tot/n:.6f} | {time.time()-t0:.2f}s")
    return func

A_model = train_true_ode(epochs=20)

# --- Evaluate & visualize ---
A_model.eval()
with torch.no_grad():
    imgs, _ = next(iter(test_loader))
    clean = imgs.to(device); noisy = add_noise(clean)
    denoised_A = odeint(A_model, noisy, torch.tezsnsor([0.,6.], device=device),
                        method='dopri5', rtol=1e-3, atol=1e-4,
                        options={'dtype': torch.float32})[-1]
    print("[A] Test MSE:", F.mse_loss(denoised_A, clean).item())

def show_triplet(clean, noisy, out, K=8, title="(A) True Neural ODE"):
    c = clean[:K].cpu().view(-1,28,28); n = noisy[:K].cpu().view(-1,28,28); o = out[:K].cpu().view(-1,28,28)
    fig,axs=plt.subplots(3,K,figsize=(1.6*K,4))
    for i in range(K):
        axs[0,i].imshow(c[i],cmap='gray'); axs[0,i].axis('off')
        axs[1,i].imshow(n[i],cmap='gray'); axs[1,i].axis('off')
        axs[2,i].imshow(o[i],cmap='gray'); axs[2,i].axis('off')
    axs[0,0].set_ylabel('Clean'); axs[1,0].set_ylabel('Noisy'); axs[2,0].set_ylabel('Denoised')
    plt.suptitle(title); plt.tight_layout(); plt.show()

show_triplet(clean, noisy, denoised_A, title="(A) True Neural ODE (fixed dim, zero drift)")
