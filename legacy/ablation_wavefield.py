import os
import json
import time
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

# 1. Global Settings
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 2. Project-relative Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_31_DIR = os.path.join(BASE_DIR, '3.1', 'model')
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, '4.3 消融实验', '波场正演消融')

FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'figures')
MODEL_DIR = os.path.join(OUTPUT_BASE_DIR, 'models')
DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, 'checkpoints')

for d in [FIG_DIR, MODEL_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. Configuration (Direct 60Hz training, no frequency progression)
CONFIG = {
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "t_range": (0.0, 1.0),
    "source_location": (0.5, 0.5, 0.12),
    "target_frequency": 60.0,
    "epochs": 2000,
    "batch_phys": 4096,
    "batch_bc": 2048,
    "batch_ic": 2048,
    "lr": 1e-4,
    "hidden_dim": 128,
    "num_layers": 5,
    "lambda_phys": 1.0,
    "lambda_ic": 10.0,
    "lambda_bc": 10.0,
}

# 4. Load velocity model
def load_velocity_model():
    print("\nLoading Overthrust velocity model...")
    vp_path = os.path.join(MODEL_31_DIR, 'vp_model.npy')
    vp = np.load(vp_path).astype(np.float32)
    print(f"Velocity shape: {vp.shape}")
    print(f"Velocity range: [{vp.min():.1f}, {vp.max():.1f}]")
    return vp

# 5. Velocity model wrapper
class VelocityModel:
    def __init__(self, vp):
        self.vp = vp
        self.nx, self.ny, self.nz = vp.shape
        self.vmin = float(vp.min())
        self.vmax = float(vp.max())

    def get_velocity(self, x, y, z):
        ix = torch.clamp((x * (self.nx - 1)).long(), 0, self.nx - 1)
        iy = torch.clamp((y * (self.ny - 1)).long(), 0, self.ny - 1)
        iz = torch.clamp((z * (self.nz - 1)).long(), 0, self.nz - 1)

        vel = self.vp[
            ix.detach().cpu().numpy(),
            iy.detach().cpu().numpy(),
            iz.detach().cpu().numpy()
        ]
        vel = torch.tensor(vel, dtype=torch.float32, device=device).view(-1, 1)

        vel = (vel - self.vmin) / (self.vmax - self.vmin + 1e-8)
        vel = 0.2 + 0.8 * vel
        return vel

# 6. SIREN layers
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first, in_features)

    def init_weights(self, is_first, in_features):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# 7. PINN model
class WavePINN(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128, num_layers=5):
        super().__init__()
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=30.0)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=30.0))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 8. Utility functions
def sample_uniform(n, require_grad=False):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    z = torch.rand(n, 1, device=device)
    t = torch.rand(n, 1, device=device)

    if require_grad:
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)

    return x, y, z, t

def ricker_wavelet(t, f0=10.0):
    pi2 = (np.pi * f0) ** 2
    return (1.0 - 2.0 * pi2 * t**2) * torch.exp(-pi2 * t**2)

# 9. Trainer (No frequency progressive strategy - direct 60Hz training)
class PINNTrainerNoProgressive:
    def __init__(self, vp):
        self.model = WavePINN(
            in_dim=4,
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"]
        ).to(device)

        self.vmodel = VelocityModel(vp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])
        self.history = {'phys_losses': [], 'ic_losses': [], 'bc_losses': [], 'total_losses': []}

    def source_term(self, x, y, z, t, f0):
        sx, sy, sz = CONFIG["source_location"]
        sigma = 0.06
        spatial = torch.exp(-((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) / (2 * sigma ** 2))
        temporal = ricker_wavelet(t - 0.08, f0=f0)
        return spatial * temporal

    def physics_loss(self, n_f, f0):
        x, y, z, t = sample_uniform(n_f, require_grad=True)
        inp = torch.cat([x, y, z, t], dim=1)
        u = self.model(inp)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True, retain_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True, retain_graph=True)[0]

        vel = self.vmodel.get_velocity(x, y, z)
        c = vel / 6000.0

        source = self.source_term(x, y, z, t, f0)
        residual = u_tt - (c ** 2) * (u_xx + u_yy + u_zz) - source
        return torch.mean(residual ** 2)

    def initial_condition_loss(self, n_ic):
        x = torch.rand(n_ic, 1, device=device)
        y = torch.rand(n_ic, 1, device=device)
        z = torch.rand(n_ic, 1, device=device)
        t = torch.zeros(n_ic, 1, device=device, requires_grad=True)

        inp = torch.cat([x, y, z, t], dim=1)
        u = self.model(inp)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        loss_u0 = torch.mean(u ** 2)
        loss_ut0 = torch.mean(u_t ** 2)
        return loss_u0 + loss_ut0

    def boundary_condition_loss(self, n_bc):
        n_side = n_bc // 6

        y = torch.rand(n_side, 1, device=device)
        z = torch.rand(n_side, 1, device=device)
        t = torch.rand(n_side, 1, device=device)
        x0 = torch.zeros_like(y)
        x1 = torch.ones_like(y)

        inp0 = torch.cat([x0, y, z, t], dim=1)
        inp1 = torch.cat([x1, y, z, t], dim=1)

        x = torch.rand(n_side, 1, device=device)
        inp2 = torch.cat([x, torch.zeros_like(x), z, t], dim=1)
        inp3 = torch.cat([x, torch.ones_like(x), z, t], dim=1)

        x = torch.rand(n_side, 1, device=device)
        y = torch.rand(n_side, 1, device=device)
        inp4 = torch.cat([x, y, torch.zeros_like(x), t], dim=1)
        inp5 = torch.cat([x, y, torch.ones_like(x), t], dim=1)

        u0 = self.model(inp0)
        u1 = self.model(inp1)
        u2 = self.model(inp2)
        u3 = self.model(inp3)
        u4 = self.model(inp4)
        u5 = self.model(inp5)

        return torch.mean(u0 ** 2) + torch.mean(u1 ** 2) + torch.mean(u2 ** 2) + torch.mean(u3 ** 2) + torch.mean(u4 ** 2) + torch.mean(u5 ** 2)

    def train(self):
        print("\n" + "=" * 80)
        print(f"Ablation Experiment: Direct Training on {int(CONFIG['target_frequency'])} Hz High-Frequency Wavefield")
        print("(Without Frequency Progressive Strategy)")
        print("=" * 80)

        start_time = time.time()
        freq = CONFIG["target_frequency"]

        for epoch in tqdm(range(CONFIG["epochs"]), desc=f"Training {int(freq)} Hz (No Progressive)"):
            self.model.train()
            self.optimizer.zero_grad()

            loss_phys = self.physics_loss(CONFIG["batch_phys"], freq)
            loss_ic = self.initial_condition_loss(CONFIG["batch_ic"])
            loss_bc = self.boundary_condition_loss(CONFIG["batch_bc"])

            loss = (
                CONFIG["lambda_phys"] * loss_phys
                + CONFIG["lambda_ic"] * loss_ic
                + CONFIG["lambda_bc"] * loss_bc
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.history['phys_losses'].append(float(loss_phys.item()))
            self.history['ic_losses'].append(float(loss_ic.item()))
            self.history['bc_losses'].append(float(loss_bc.item()))
            self.history['total_losses'].append(float(loss.item()))

            if (epoch + 1) % 100 == 0:
                print(
                    f"[{int(freq)} Hz | Epoch {epoch+1:4d}] "
                    f"Phys={loss_phys.item():.6e} | "
                    f"IC={loss_ic.item():.6e} | "
                    f"BC={loss_bc.item():.6e} | "
                    f"Total={loss.item():.6e}"
                )

        elapsed = time.time() - start_time

        freq_tag = f"{int(freq)}Hz_no_progressive"
        np.save(os.path.join(DATA_DIR, f"loss_phys_{freq_tag}.npy"), np.array(self.history['phys_losses']))
        np.save(os.path.join(DATA_DIR, f"loss_ic_{freq_tag}.npy"), np.array(self.history['ic_losses']))
        np.save(os.path.join(DATA_DIR, f"loss_bc_{freq_tag}.npy"), np.array(self.history['bc_losses']))
        np.save(os.path.join(DATA_DIR, f"loss_total_{freq_tag}.npy"), np.array(self.history['total_losses']))

        model_path = os.path.join(MODEL_DIR, f"model_{freq_tag}.pth")
        torch.save(self.model.state_dict(), model_path)

        report = {
            "experiment": "Ablation - No Frequency Progressive",
            "frequency_hz": float(freq),
            "final_physics_loss": float(self.history['phys_losses'][-1]),
            "final_ic_loss": float(self.history['ic_losses'][-1]),
            "final_bc_loss": float(self.history['bc_losses'][-1]),
            "final_total_loss": float(self.history['total_losses'][-1]),
            "training_time_sec": float(elapsed),
            "model_path": model_path,
            "epochs": CONFIG["epochs"]
        }
        with open(os.path.join(DATA_DIR, f"report_{freq_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n" + "-" * 80)
        print(f"Ablation Experiment (No Frequency Progressive) Completed")
        print(f"Final Physics Loss: {self.history['phys_losses'][-1]:.6e}")
        print(f"Final IC Loss: {self.history['ic_losses'][-1]:.6e}")
        print(f"Final BC Loss: {self.history['bc_losses'][-1]:.6e}")
        print(f"Final Total Loss: {self.history['total_losses'][-1]:.6e}")
        print(f"Training Time: {elapsed/60:.2f} min")
        print(f"Model Saved To: {model_path}")
        print("-" * 80)

        return self.history

    def plot_losses(self, save_path):
        plt.figure(figsize=(8, 6), dpi=300)
        plt.semilogy(self.history['phys_losses'], label="Physics Loss", linewidth=2)
        plt.semilogy(self.history['ic_losses'], label="Initial Condition Loss", linewidth=2)
        plt.semilogy(self.history['bc_losses'], label="Boundary Condition Loss", linewidth=2)
        plt.semilogy(self.history['total_losses'], label="Total Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Ablation Experiment: Direct 60Hz Training (No Progressive Strategy)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_slice(self, save_path):
        self.model.eval()
        with torch.no_grad():
            nx, ny = 120, 120
            x = torch.linspace(0, 1, nx, device=device)
            y = torch.linspace(0, 1, ny, device=device)
            X, Y = torch.meshgrid(x, y, indexing="ij")
            Z = torch.full_like(X, 0.5)
            T = torch.full_like(X, 0.5)
            inp = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1), T.reshape(-1)], dim=1)
            U = self.model(inp).reshape(nx, ny).cpu().numpy()

        plt.figure(figsize=(7, 6), dpi=300)
        plt.imshow(U, cmap="seismic", origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        plt.colorbar(label="Amplitude")
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.title(f"Wavefield Slice at z=0.5, t=0.5 (Direct 60Hz Training)", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.model.train()


# 10. Main
def main():
    print("=" * 80)
    print("Ablation Experiment: Wavefield Forward Modeling Without Frequency Progressive Strategy")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    vp = load_velocity_model()
    trainer = PINNTrainerNoProgressive(vp)
    history = trainer.train()

    freq_tag = f"{int(CONFIG['target_frequency'])}Hz_no_progressive"
    trainer.plot_losses(save_path=os.path.join(FIG_DIR, f"loss_curves_{freq_tag}.png"))
    trainer.plot_slice(save_path=os.path.join(FIG_DIR, f"wavefield_slice_{freq_tag}.png"))

    with open(os.path.join(OUTPUT_BASE_DIR, 'experiment_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Ablation Experiment: No Frequency Progressive Strategy\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Target frequency: {CONFIG['target_frequency']} Hz (direct training)\n")
        f.write(f"  Epochs: {CONFIG['epochs']}\n")
        f.write(f"  Hidden dimension: {CONFIG['hidden_dim']}\n")
        f.write(f"  Number of layers: {CONFIG['num_layers']}\n")
        f.write(f"  Learning rate: {CONFIG['lr']}\n")
        f.write(f"  lambda_phys: {CONFIG['lambda_phys']}\n")
        f.write(f"  lambda_ic: {CONFIG['lambda_ic']}\n")
        f.write(f"  lambda_bc: {CONFIG['lambda_bc']}\n\n")
        f.write("Results:\n")
        f.write(f"  Final Physics Loss: {history['phys_losses'][-1]:.6e}\n")
        f.write(f"  Final IC Loss: {history['ic_losses'][-1]:.6e}\n")
        f.write(f"  Final BC Loss: {history['bc_losses'][-1]:.6e}\n")
        f.write(f"  Final Total Loss: {history['total_losses'][-1]:.6e}\n")

    print("\n" + "=" * 60)
    print("Ablation Experiment Completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data saved to: {DATA_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"Model saved to: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
