import os
import json
import math
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
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_DIR, '3.2')
MODEL_31_DIR = os.path.join(PROJECT_DIR, '3.1', 'model')

FIG_DIR = os.path.join(BASE_DIR, 'figures')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for d in [FIG_DIR, MODEL_DIR, DATA_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. Configuration
CONFIG = {
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "t_range": (0.0, 1.0),
    "source_location": (0.5, 0.5, 0.12),
    "source_frequency": 10.0,
    "frequencies": [5.0, 10.0, 15.0, 20.0, 30.0],
    "epochs_per_freq": 2000,
    "batch_phys": 4096,
    "batch_bc": 2048,
    "batch_ic": 2048,
    "lr": 1e-4,
    "hidden_dim": 128,
    "num_layers": 5,
    "adaptive_resample_every": 100,
    "adaptive_candidate_size": 20000,
    "adaptive_keep_topk": 4096,
    "lambda_phys": 1.0,
    "lambda_ic": 10.0,
    "lambda_bc": 10.0,
    "lambda_src": 1.0,
    "save_every_epoch": 200
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

# 9. Trainer
class PINNTrainer:
    def __init__(self, vp):
        self.model = WavePINN(
            in_dim=4,
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"]
        ).to(device)

        self.vmodel = VelocityModel(vp)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])

        self.history = {}
        self.current_freq_idx = 0
        self.current_epoch_in_stage = 0

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

    def adaptive_residual_sampling(self, f0):
        self.model.eval()

        n_cand = CONFIG["adaptive_candidate_size"]
        x, y, z, t = sample_uniform(n_cand, require_grad=True)
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
        residual = (u_tt - (c ** 2) * (u_xx + u_yy + u_zz) - source).detach().abs().squeeze()

        topk = min(CONFIG["adaptive_keep_topk"], residual.numel())
        _, idx = torch.topk(residual, topk)

        x_new = x[idx].detach()
        y_new = y[idx].detach()
        z_new = z[idx].detach()
        t_new = t[idx].detach()

        self.model.train()
        return x_new, y_new, z_new, t_new

    def save_checkpoint(self, freq_idx, epoch, freq, history):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_freq_{int(freq)}hz_epoch_{epoch}.pth")
        state = {
            "freq_idx": freq_idx,
            "epoch": epoch,
            "freq": freq,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history
        }
        torch.save(state, ckpt_path)

        latest_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        torch.save(state, latest_path)

        print(f"\nCheckpoint saved: {ckpt_path}")

    def load_latest_checkpoint(self):
        latest_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
        if not os.path.exists(latest_path):
            return None

        ckpt = torch.load(latest_path, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_freq_idx = ckpt.get("freq_idx", 0)
        self.current_epoch_in_stage = ckpt.get("epoch", 0)
        self.history = ckpt.get("history", {})

        print(f"Loaded checkpoint from freq_idx={self.current_freq_idx}, epoch={self.current_epoch_in_stage}")
        return ckpt

    def plot_losses(self, freq, phys_hist, ic_hist, bc_hist, total_hist, save_path):
        plt.figure(figsize=(8, 6))
        plt.semilogy(phys_hist, label="Physics Loss", linewidth=2)
        plt.semilogy(ic_hist, label="Initial Condition Loss", linewidth=2)
        plt.semilogy(bc_hist, label="Boundary Condition Loss", linewidth=2)
        plt.semilogy(total_hist, label="Total Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Training Loss Curves at {int(freq)} Hz", fontsize=14)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_slice(self, freq, save_path):
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

        plt.figure(figsize=(7, 6))
        plt.imshow(U, cmap="seismic", origin="lower", extent=[0, 1, 0, 1], aspect="auto")
        plt.colorbar(label="Amplitude")
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.title(f"Predicted Wavefield Slice at z=0.5, t=0.5 ({int(freq)} Hz)", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.model.train()

    def train_frequency_stage(self, freq_idx, freq):
        print("\n" + "=" * 80)
        print(f"START TRAINING FREQUENCY STAGE: {int(freq)} Hz")
        print("=" * 80)

        phys_hist, ic_hist, bc_hist, total_hist = [], [], [], []

        start_time = time.time()

        for epoch in tqdm(range(CONFIG["epochs_per_freq"]), desc=f"Training {int(freq)} Hz"):
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

            phys_hist.append(float(loss_phys.item()))
            ic_hist.append(float(loss_ic.item()))
            bc_hist.append(float(loss_bc.item()))
            total_hist.append(float(loss.item()))

            if (epoch + 1) % 100 == 0:
                print(
                    f"[{int(freq)} Hz | Epoch {epoch+1:4d}] "
                    f"Phys={loss_phys.item():.6e} | "
                    f"IC={loss_ic.item():.6e} | "
                    f"BC={loss_bc.item():.6e} | "
                    f"Total={loss.item():.6e}"
                )

            if (epoch + 1) % CONFIG["adaptive_resample_every"] == 0:
                print(f"Adaptive resampling at epoch {epoch+1}")
                _ = self.adaptive_residual_sampling(freq)

            if (epoch + 1) % CONFIG["save_every_epoch"] == 0:
                self.save_checkpoint(freq_idx, epoch + 1, freq, {
                    "phys": phys_hist,
                    "ic": ic_hist,
                    "bc": bc_hist,
                    "total": total_hist
                })

        elapsed = time.time() - start_time

        freq_tag = f"{int(freq)}Hz"
        np.save(os.path.join(DATA_DIR, f"loss_phys_{freq_tag}.npy"), np.array(phys_hist))
        np.save(os.path.join(DATA_DIR, f"loss_ic_{freq_tag}.npy"), np.array(ic_hist))
        np.save(os.path.join(DATA_DIR, f"loss_bc_{freq_tag}.npy"), np.array(bc_hist))
        np.save(os.path.join(DATA_DIR, f"loss_total_{freq_tag}.npy"), np.array(total_hist))

        self.plot_losses(
            freq,
            phys_hist,
            ic_hist,
            bc_hist,
            total_hist,
            os.path.join(FIG_DIR, f"loss_curves_{freq_tag}.png")
        )
        self.plot_slice(
            freq,
            os.path.join(FIG_DIR, f"wavefield_slice_{freq_tag}.png")
        )

        model_path = os.path.join(MODEL_DIR, f"model_{freq_tag}.pth")
        torch.save(self.model.state_dict(), model_path)

        report = {
            "frequency_hz": float(freq),
            "final_physics_loss": float(phys_hist[-1]),
            "final_ic_loss": float(ic_hist[-1]),
            "final_bc_loss": float(bc_hist[-1]),
            "final_total_loss": float(total_hist[-1]),
            "training_time_sec": float(elapsed),
            "model_path": model_path
        }
        with open(os.path.join(DATA_DIR, f"report_{freq_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n" + "-" * 80)
        print(f"Finished {int(freq)} Hz stage.")
        print(f"Physics Loss      : {phys_hist[-1]:.6e}")
        print(f"Initial Loss      : {ic_hist[-1]:.6e}")
        print(f"Boundary Loss     : {bc_hist[-1]:.6e}")
        print(f"Final Total Loss   : {total_hist[-1]:.6e}")
        print(f"Stage Time         : {elapsed/60:.2f} min")
        print(f"Model Saved To     : {model_path}")
        print(f"Report Saved To    : {os.path.join(DATA_DIR, f'report_{freq_tag}.json')}")
        print("-" * 80)

        return phys_hist, ic_hist, bc_hist, total_hist

# 10. Main
def main():
    print("=" * 80)
    print("REAL 3D PINN WAVE EQUATION")
    print("Curriculum Learning + Frequency Progression + Adaptive Sampling")
    print("=" * 80)

    vp = load_velocity_model()
    trainer = PINNTrainer(vp)

    latest = trainer.load_latest_checkpoint()
    if latest is not None:
        print("Checkpoint loaded successfully.")
    else:
        print("No checkpoint found. Start from scratch.")

    summary = {}

    for freq_idx, freq in enumerate(CONFIG["frequencies"]):
        phys_hist, ic_hist, bc_hist, total_hist = trainer.train_frequency_stage(freq_idx, freq)
        summary[f"{int(freq)}Hz"] = {
            "final_physics_loss": float(phys_hist[-1]),
            "final_ic_loss": float(ic_hist[-1]),
            "final_bc_loss": float(bc_hist[-1]),
            "final_total_loss": float(total_hist[-1]),
        }

        print(f"\n>>> Summary for {int(freq)} Hz")
        print(f"Physics Loss  : {phys_hist[-1]:.6e}")
        print(f"IC Loss       : {ic_hist[-1]:.6e}")
        print(f"BC Loss       : {bc_hist[-1]:.6e}")
        print(f"Total Loss    : {total_hist[-1]:.6e}")
        print("=" * 80)

    summary_path = os.path.join(DATA_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining completed successfully.")
    print(f"Training summary saved to: {summary_path}")
    print("All frequency-stage results have been saved.")

if __name__ == "__main__":
    main()
