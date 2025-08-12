import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ---- Noise ----
def add_noise(img, noise_factor=0.3):
    noisy = img + noise_factor * torch.randn_like(img)
    return torch.clamp(noisy, 0., 1.)

# ---- Data ----
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=128, shuffle=False)

# ---- Model ----
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

# ---- Training setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
noise_factor = 0.3
epochs = 10  # change as you like

def evaluate(loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            noisy = add_noise(imgs, noise_factor)
            outs = model(noisy)
            loss = criterion(outs, imgs)
            bsz = imgs.size(0)
            total += loss.item() * bsz
            count += bsz
    return total / count

# ---- Train ----
for epoch in range(epochs):
    model.train()
    train_total, train_count = 0.0, 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs, noise_factor)

        outputs = model(noisy_imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bsz = imgs.size(0)
        train_total += loss.item() * bsz
        train_count += bsz

    train_avg = train_total / train_count
    test_avg  = evaluate(test_loader)
    print(f"Epoch {epoch+1:02d} | train_MSE={train_avg:.4f} | test_MSE={test_avg:.4f}")