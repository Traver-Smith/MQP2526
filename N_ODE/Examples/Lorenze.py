import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import CheckButtons
import time

#-----------------------------------------------------------
#Goals:
#   1. Train a NODE to match the Lorenz attractor
#   2. Compare NODE's ability to match discrete and continuous versions of Lorenz
#       - Generate a Euler step Lorenz approx
#       - Solve the actual continuous ODE version of Lorenz and sample
#       - Train one NODE on the continuous d/dt field
#       - Train the other on the discrete finite-difference field
#       - Compare performance
#-----------------------------------------------------------

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

#-------------------------------------------------------
# Lorenz ODE (continuous)
#-------------------------------------------------------
def lorenz_ode(t, state, sigma=10., rho=28., beta=8/3.):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz])

# Vectorized Lorenz derivative (batch-safe): works for [..., 3]
def lorenz_vec(states, sigma=10., rho=28., beta=8/3.):
    x, y, z = states[..., 0], states[..., 1], states[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=-1)

#-------------------------------------------------------
# Data generation params
#-------------------------------------------------------
dt = 0.01
T = 4000  # total steps
t = torch.linspace(0., T * dt, T).cpu()
n_traj = 100
y0s = torch.randn(n_traj, 3) * 5  # CPU on purpose for odeint stability

#-------------------------------------------------------
# 1) Generate high-precision continuous trajectories (with progress)
#-------------------------------------------------------
true_trajs = []
start = time.time()
for i in range(n_traj):
    if i > 0 and i % max(1, n_traj // 10) == 0:
        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-8)
        eta = (n_traj - i) / max(rate, 1e-8)
        print(f"[Gen:Continuous] {i}/{n_traj} done | {rate:.2f} traj/s | ETA {eta:.1f}s")
    traj_i = odeint(lorenz_ode, y0s[i], t, method='dopri5')
    true_trajs.append(traj_i)
elapsed = time.time() - start
print(f"[Gen:Continuous] Done {n_traj}/{n_traj} in {elapsed:.1f}s "
      f"({n_traj/max(elapsed,1e-8):.2f} traj/s)")

true_traj = torch.stack(true_trajs, dim=1)  # [T, n_traj, 3]
dtrue = lorenz_vec(true_traj)               # [T, n_traj, 3]

# Flat training data (continuous)
X_cont = true_traj[:-1].reshape(-1, 3)
Ydot_cont = dtrue[:-1].reshape(-1, 3)

#-------------------------------------------------------
# 2) Generate discrete (Euler) trajectories (with progress)
#-------------------------------------------------------
def lorenz_batch(states, sigma=10., rho=28., beta=8/3.):
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz], dim=1)

starts = (torch.randn(n_traj, 3) * 5).to(device)
states = starts.clone()
all_traj = []

start = time.time()
for k in range(T):
    if k > 0 and k % max(1, T // 10) == 0:
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - start
        rate = k / max(elapsed, 1e-8)
        eta = (T - k) / max(rate, 1e-8)
        print(f"[Gen:Discrete] {k}/{T} steps | {rate:.1f} steps/s | ETA {eta:.1f}s")
    states = states + dt * lorenz_batch(states)
    all_traj.append(states.clone())
if device == "mps":
    torch.mps.synchronize()
elapsed = time.time() - start
print(f"[Gen:Discrete] Done {T}/{T} in {elapsed:.1f}s "
      f"({T/max(elapsed,1e-8):.1f} steps/s)")

discrete_traj = torch.stack(all_traj, dim=0)  # [T, n_traj, 3]
X_disc = discrete_traj[:-1].reshape(-1, 3)
Y_disc = discrete_traj[1:].reshape(-1, 3)
Ydot_disc = (Y_disc - X_disc) / dt

#-------------------------------------------------------
# Normalize
#-------------------------------------------------------
mean_y_cont, std_y_cont = X_cont.mean(0), X_cont.std(0)
mean_dy_cont, std_dy_cont = Ydot_cont.mean(0), Ydot_cont.std(0)
Xc = (X_cont - mean_y_cont) / std_y_cont
Yc = (Ydot_cont - mean_dy_cont) / std_dy_cont

mean_y_disc, std_y_disc = X_disc.mean(0), X_disc.std(0)
mean_dy_disc, std_dy_disc = Ydot_disc.mean(0), Ydot_disc.std(0)
Xd = (X_disc - mean_y_disc) / std_y_disc
Yd = (Ydot_disc - mean_dy_disc) / std_dy_disc

# Targets for the map: predict next *normalized* state directly
Ynext_norm_disc = (Y_disc - mean_y_disc) / std_y_disc  # shape [N, 3]
Xmap = Xd  # inputs already normalized
Ymap = Ynext_norm_disc

#-------------------------------------------------------
# ODE Model
#-------------------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
    def forward(self, t, y):
        return self.net(y)

#-------------------------------------------------------
# Train ODEs
#-------------------------------------------------------
def train_model(X, Y, label, epochs=3000, batch_size=1024):
    model = ODEFunc().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = X.to(device); Y = Y.to(device)

    tick = time.time()
    for epoch in range(epochs + 1):
        idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
        x_batch = X[idx]
        y_true = Y[idx]

        y_pred = model(0, x_batch)
        loss = (y_true - y_pred).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0:
            if device == "mps": torch.mps.synchronize()
            tock = time.time()
            print(f"[{label}] Epoch {epoch:4d} | Loss {loss.item():.6f} | "
                  f"{(tock - tick):.3f} s / 100 epochs")
            tick = time.time()
    return model

print("\n=== Training (continuous) ===")
f_cont = train_model(Xc, Yc, "Continuous", epochs=3000, batch_size=1024)

print("\n=== Training (discrete) ===")
f_disc = train_model(Xd, Yd, "Discrete", epochs=3000, batch_size=1024)

#-------------------------------------------------------
# Neural Iterative Map (x_{n+1} = f_map(x_n) on normalized states)
#-------------------------------------------------------
class MapNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)
        )
    def forward(self, x):  # x is normalized state at step n
        return self.net(x) # returns normalized state at step n+1

# Train Neural Map on discrete pairs (x_n -> x_{n+1})
f_map = MapNet().to(device)
opt_map = torch.optim.Adam(f_map.parameters(), lr=1e-3)

Xmap = Xmap.to(device); Ymap = Ymap.to(device)

batch_size = 1024
epochs = 3000
tick = time.time()
for epoch in range(epochs + 1):
    idx = torch.randint(0, Xmap.shape[0], (batch_size,), device=device)
    x_batch = Xmap[idx]       # normalized x_n
    y_next_true = Ymap[idx]   # normalized x_{n+1}

    y_next_pred = f_map(x_batch)
    loss = (y_next_true - y_next_pred).pow(2).mean()

    opt_map.zero_grad()
    loss.backward()
    opt_map.step()

    if epoch % 100 == 0:
        if device == "mps": torch.mps.synchronize()
        tock = time.time()
        print(f"[Map] Epoch {epoch:4d} | Loss {loss.item():.6f} | "
              f"{(tock - tick):.3f} s / 100 epochs")
        tick = time.time()

#-------------------------------------------------------
# Integrate/iterate learned systems from the SAME initial state
#-------------------------------------------------------
y0_cpu = y0s[0].clone().cpu()

# Normalize in the coordinate systems each model expects
y0_norm_cont = (y0_cpu - mean_y_cont.cpu()) / std_y_cont.cpu()
y0_norm_disc = (y0_cpu - mean_y_disc.cpu()) / std_y_disc.cpu()

t_test = t  # CPU
f_cont_cpu = f_cont.to("cpu").eval()
f_disc_cpu = f_disc.to("cpu").eval()

with torch.no_grad():
    # Map rollout in normalized (discrete) space
    f_map_cpu = f_map.to("cpu").eval()
    y_map = (y0_cpu - mean_y_disc.cpu()) / std_y_disc.cpu()
    traj_list = [y_map.unsqueeze(0)]
    for _ in range(T-1):
        y_map = f_map_cpu(y_map)
        traj_list.append(y_map.unsqueeze(0))
    pred_map = torch.cat(traj_list, dim=0)                         # [T, 3] normalized
    pred_map = pred_map * std_y_disc.cpu() + mean_y_disc.cpu()     # denormalize

with torch.no_grad():
    pred_cont = odeint(f_cont_cpu, y0_norm_cont, t_test, method='dopri5')
    pred_disc = odeint(f_disc_cpu, y0_norm_disc, t_test, method='dopri5')
    pred_cont = pred_cont * std_y_cont.cpu() + mean_y_cont.cpu()
    pred_disc = pred_disc * std_y_disc.cpu() + mean_y_disc.cpu()

# Ground truth trajectory from same y0
true_traj_single = odeint(lorenz_ode, y0_cpu, t_test, method='dopri5')

#-------------------------------------------------------
# Final test losses (MSE on positions over the whole path)
#-------------------------------------------------------
true_np = true_traj_single.numpy()
cont_np = pred_cont.numpy()
disc_np = pred_disc.numpy()
map_np  = pred_map.numpy()

mse_cont = np.mean((true_np - cont_np) ** 2)
mse_disc = np.mean((true_np - disc_np) ** 2)
mse_map  = np.mean((true_np - map_np)  ** 2)
print(f"\n[TEST] Position MSE | Continuous: {mse_cont:.6f} | Discrete: {mse_disc:.6f} | Map: {mse_map:.6f}")

#-------------------------------------------------------
# Speeds over time for the four curves
#-------------------------------------------------------
def speeds_from_traj(arr, dt):
    vel = np.diff(arr, axis=0) / dt
    spd = np.linalg.norm(vel, axis=1)
    return np.concatenate([spd, spd[-1:]], axis=0)

speed_true = speeds_from_traj(true_np, dt)
speed_cont = speeds_from_traj(cont_np, dt)
speed_disc = speeds_from_traj(disc_np, dt)
speed_map  = speeds_from_traj(map_np,  dt)
tt = t_test.numpy()

#-------------------------------------------------------
# Visualization: 3 panels + toggles (now includes Map)
#-------------------------------------------------------
fig = plt.figure(figsize=(13, 8))

# 3D Trajectory
ax3d = fig.add_subplot(2, 2, 1, projection='3d')
true_3d, = ax3d.plot(true_np[:, 0], true_np[:, 1], true_np[:, 2], lw=0.7, label='True Lorenz')
cont_3d, = ax3d.plot(cont_np[:, 0], cont_np[:, 1], cont_np[:, 2], lw=0.7, label='Neural ODE (Continuous)')
disc_3d, = ax3d.plot(disc_np[:, 0], disc_np[:, 1], disc_np[:, 2], lw=0.7, label='Neural ODE (Discrete)')
map_3d,  = ax3d.plot(map_np[:,  0], map_np[:,  1], map_np[:,  2],  lw=0.7, label='Neural Map (Discrete)')
ax3d.set_title("3D Trajectories")

# 2D x–z projection
ax2d = fig.add_subplot(2, 2, 2)
true_2d, = ax2d.plot(true_np[:, 0], true_np[:, 2], lw=0.7, label='True Lorenz')
cont_2d, = ax2d.plot(cont_np[:, 0], cont_np[:, 2], lw=0.7, label='Neural ODE (Continuous)')
disc_2d, = ax2d.plot(disc_np[:, 0], disc_np[:, 2], lw=0.7, label='Neural ODE (Discrete)')
# FIX: map in 2D should be (x, z), not (x, y, z)
map_2d,  = ax2d.plot(map_np[:,  0], map_np[:,  2],  lw=0.7, label='Neural Map (Discrete)')
ax2d.set_xlabel('x'); ax2d.set_ylabel('z'); ax2d.set_title('x–z Projection')

# Speed vs time
axspd = fig.add_subplot(2, 1, 2)
true_spd_line, = axspd.plot(tt, speed_true, lw=0.9, label='True speed')
cont_spd_line, = axspd.plot(tt, speed_cont, lw=0.9, label='Continuous speed')
disc_spd_line, = axspd.plot(tt, speed_disc, lw=0.9, label='Discrete speed')
map_spd_line,  = axspd.plot(tt, speed_map,  lw=0.9, label='Neural Map speed')
axspd.set_xlabel('time'); axspd.set_ylabel('speed'); axspd.set_title('Speed vs Time')
axspd.legend(loc='upper right')

# Shared legend + checkboxes (now includes Map)
handles = [true_3d, cont_3d, disc_3d, map_3d]
labels  = ['True Lorenz', 'Continuous', 'Discrete', 'Map']

rax = plt.axes([0.015, 0.42, 0.16, 0.20])
check = CheckButtons(rax, labels, [h.get_visible() for h in handles])

def set_visible(line_list, visible):
    for ln in line_list:
        ln.set_visible(visible)

# Map each label to all corresponding lines across subplots
lines_map = {
    'True Lorenz': [true_3d, true_2d, true_spd_line],
    'Continuous':  [cont_3d, cont_2d, cont_spd_line],
    'Discrete':    [disc_3d, disc_2d, disc_spd_line],
    'Map':         [map_3d,  map_2d,  map_spd_line],
}

def on_check(label):
    target_lines = lines_map[label]
    new_visible = not target_lines[0].get_visible()
    set_visible(target_lines, new_visible)
    plt.draw()

check.on_clicked(on_check)


#-------------------------------------------------------
# Add 4th subplot: per-step RMS state error
#-------------------------------------------------------
axerr = fig.add_subplot(2, 2, 4)
err_cont = np.sqrt(np.mean((true_np - cont_np) ** 2, axis=1))
err_disc = np.sqrt(np.mean((true_np - disc_np) ** 2, axis=1))
err_map  = np.sqrt(np.mean((true_np - map_np)  ** 2, axis=1))

axerr.plot(tt, err_cont, color='green', lw=0.9, label='Continuous error')
axerr.plot(tt, err_disc, color='orange', lw=0.9, label='Discrete error')
axerr.plot(tt, err_map,  color='purple', lw=0.9, label='Map error')
axerr.set_xlabel('time'); axerr.set_ylabel('RMS state error')
axerr.set_title('Divergence Over Time')
axerr.legend(loc='upper left')

# Adjust layout to fit 4 panels
plt.tight_layout(rect=[0.19, 0.05, 1, 0.98])
plt.show()
