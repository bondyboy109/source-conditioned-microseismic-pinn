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

# 1. Global settings
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# 2. Project-relative path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_32_DIR = os.path.join(BASE_DIR, '3.2', 'models')
WAVEFORM_DIR = os.path.join(BASE_DIR, '3.3', 'waveform_data')
FIG_DIR = os.path.join(BASE_DIR, '3.3', 'figures')
OUTPUT_DIR = os.path.join(BASE_DIR, '3.3', 'results')

for d in [FIG_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# 3. Configuration (consistent with Section 3.2)
CONFIG = {
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "t_range": (0.0, 1.0),
    "frequencies": [5.0, 10.0, 15.0, 20.0, 30.0],
    "default_freq": 20.0,
    "grid_search_resolution": {
        "x": 20,
        "y": 20,
        "z": 15,
        "t": 15
    },
    "local_optimization_iterations": 300,
    "local_optimization_lr": 0.005,
    "vp_min": 2000.0,
    "vp_max": 6000.0,
    "avg_vp": 4000.0,
    "search_bounds": {
        "x": (2000.0, 14000.0),
        "y": (2000.0, 14000.0),
        "z": (100.0, 1500.0),
        "t": (0.0, 0.5)
    }
}


# 4. SIREN layer (consistent with Section 3.2)
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


# 5. PINN wavefield model (consistent with Section 3.2)
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


def load_pretrained_model(freq_hz, model_dir, device):
    freq_tag = f"{int(freq_hz)}Hz"
    model_path = os.path.join(model_dir, f"model_{freq_tag}.pth")

    print(f"\nLoading pretrained model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = WavePINN(in_dim=4, hidden_dim=128, num_layers=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"Loaded {freq_tag} model")
    return model


# 6. Waveform data loading
def load_waveform_data(data_dir):
    print("\nLoading waveform data...")

    npz_path = os.path.join(data_dir, 'npr3_gathers_processed.npz')

    if not os.path.exists(npz_path):
        print(f"Warning: Data file not found - {npz_path}")
        print("Using synthetic data for testing...")
        return generate_synthetic_waveform_data()

    data = np.load(npz_path)
    waveform_data = data['data']
    times = data['times']

    n_receivers = waveform_data.shape[0]
    receiver_coords = generate_receiver_coordinates(n_receivers)

    print(f"Data loaded successfully")
    print(f"  Shape: {waveform_data.shape}")
    print(f"  Time range: [{times[0]:.4f}, {times[-1]:.4f}] s")
    print(f"  Receivers: {n_receivers}")

    return waveform_data, times, receiver_coords


def generate_receiver_coordinates(n_receivers):
    x_start, x_end = 0.0, 16000.0
    y_center = 8000.0
    z_surface = 100.0

    receiver_coords = np.zeros((n_receivers, 3))
    receiver_coords[:, 0] = np.linspace(x_start, x_end, n_receivers)
    receiver_coords[:, 1] = y_center
    receiver_coords[:, 2] = z_surface

    return receiver_coords


def generate_synthetic_waveform_data():
    print("Generating synthetic waveform data...")

    n_receivers = 100
    n_samples = 500
    dt = 0.002
    times = np.arange(n_samples) * dt

    true_source = (7000.0, 8000.0, 500.0, 0.2)
    receiver_coords = generate_receiver_coordinates(n_receivers)

    waveform_data = np.zeros((n_receivers, n_samples))
    f0 = 20.0

    for i in range(n_receivers):
        dx = receiver_coords[i, 0] - true_source[0]
        dy = receiver_coords[i, 1] - true_source[1]
        dz = receiver_coords[i, 2] - true_source[2]
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        travel_time = distance / CONFIG["avg_vp"]
        arrival_time = true_source[3] + travel_time

        for j, t in enumerate(times):
            if t >= arrival_time:
                tau = np.pi * f0 * (t - arrival_time)
                waveform_data[i, j] = (1 - 2 * tau ** 2) * np.exp(-tau ** 2)

    noise_level = 0.05
    waveform_data += noise_level * np.random.randn(*waveform_data.shape)

    print(f"Synthetic data generated")
    print(f"  True source: ({true_source[0]}, {true_source[1]}, {true_source[2]}) m, t={true_source[3]} s")

    return waveform_data, times, receiver_coords


# 7. Coordinate transformation
def real_to_normalized_coords(coords, bounds=None):
    if bounds is None:
        bounds = CONFIG["search_bounds"]

    x_norm = (coords[0] - bounds["x"][0]) / (bounds["x"][1] - bounds["x"][0])
    y_norm = (coords[1] - bounds["y"][0]) / (bounds["y"][1] - bounds["y"][0])
    z_norm = (coords[2] - bounds["z"][0]) / (bounds["z"][1] - bounds["z"][0])
    t_norm = coords[3] / bounds["t"][1]

    return np.array([x_norm, y_norm, z_norm, t_norm])


def normalized_to_real_coords(coords_norm, bounds=None):
    if bounds is None:
        bounds = CONFIG["search_bounds"]

    x_real = bounds["x"][0] + coords_norm[0] * (bounds["x"][1] - bounds["x"][0])
    y_real = bounds["y"][0] + coords_norm[1] * (bounds["y"][1] - bounds["y"][0])
    z_real = bounds["z"][0] + coords_norm[2] * (bounds["z"][1] - bounds["z"][0])
    t_real = coords_norm[3] * bounds["t"][1]

    return np.array([x_real, y_real, z_real, t_real])


# 8. Source inversion
class SourceInversion:
    def __init__(self, wavefield_model, device, freq_hz=20.0):
        self.model = wavefield_model
        self.device = device
        self.freq_hz = freq_hz
        self.model.eval()

    def compute_wavefield_fast(self, receiver_coords_real, times_real, source_params_real):
        xs, ys, zs, ts = source_params_real
        n_receivers = len(receiver_coords_real)
        n_times = len(times_real)

        dx = receiver_coords_real[:, 0] - xs
        dy = receiver_coords_real[:, 1] - ys
        dz = receiver_coords_real[:, 2] - zs
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        travel_times = distances / CONFIG["avg_vp"]

        u_pred = np.zeros((n_receivers, n_times))
        f0 = self.freq_hz

        for i in range(n_receivers):
            arrival_time = ts + travel_times[i]
            for j, t in enumerate(times_real):
                if t >= arrival_time:
                    dt_arrival = t - arrival_time
                    tau = np.pi * f0 * dt_arrival
                    u_pred[i, j] = (1 - 2 * tau ** 2) * np.exp(-tau ** 2)

        return u_pred

    def grid_search(self, receiver_coords_real, times_real, u_obs):
        print("\n" + "=" * 60)
        print("Stage 1: Grid Search Coarse Location")
        print("=" * 60)

        bounds = CONFIG["search_bounds"]
        res = CONFIG["grid_search_resolution"]

        x_vals = np.linspace(bounds["x"][0], bounds["x"][1], res["x"])
        y_vals = np.linspace(bounds["y"][0], bounds["y"][1], res["y"])
        z_vals = np.linspace(bounds["z"][0], bounds["z"][1], res["z"])
        t_vals = np.linspace(bounds["t"][0], bounds["t"][1], res["t"])

        n_receivers = len(receiver_coords_real)
        n_times = len(times_real)
        u_obs_2d = u_obs.reshape(n_receivers, n_times)

        total_combinations = len(x_vals) * len(y_vals) * len(z_vals) * len(t_vals)
        print(f"  Search space: {total_combinations} combinations")

        best_loss = float('inf')
        best_params = None

        pbar = tqdm(total=total_combinations, desc="Grid search")

        for xs in x_vals:
            for ys in y_vals:
                for zs in z_vals:
                    for ts in t_vals:
                        u_pred = self.compute_wavefield_fast(
                            receiver_coords_real, times_real, (xs, ys, zs, ts)
                        )

                        valid_indices = times_real >= ts
                        if valid_indices.sum() > 0:
                            loss = np.mean((u_pred[:, valid_indices] - u_obs_2d[:, valid_indices]) ** 2)
                        else:
                            loss = float('inf')

                        if loss < best_loss:
                            best_loss = loss
                            best_params = (xs, ys, zs, ts)

                        pbar.update(1)
                        pbar.set_postfix({'best_loss': f'{best_loss:.6f}'})

        pbar.close()

        print(f"\n  Grid search result:")
        print(f"    X = {best_params[0]:.1f} m")
        print(f"    Y = {best_params[1]:.1f} m")
        print(f"    Z = {best_params[2]:.1f} m")
        print(f"    T = {best_params[3]:.4f} s")
        print(f"    Loss = {best_loss:.6f}")

        return best_params, best_loss

    def local_optimization(self, receiver_coords_real, times_real, u_obs,
                           init_params_real, n_iterations=300, lr=0.005):
        print("\n" + "=" * 60)
        print("Stage 2: Local Optimization Fine Location")
        print("=" * 60)

        init_params_norm = real_to_normalized_coords(init_params_real)
        bounds = CONFIG["search_bounds"]

        receiver_coords_norm = np.zeros_like(receiver_coords_real)
        receiver_coords_norm[:, 0] = (receiver_coords_real[:, 0] - bounds["x"][0]) / (bounds["x"][1] - bounds["x"][0])
        receiver_coords_norm[:, 1] = (receiver_coords_real[:, 1] - bounds["y"][0]) / (bounds["y"][1] - bounds["y"][0])
        receiver_coords_norm[:, 2] = (receiver_coords_real[:, 2] - bounds["z"][0]) / (bounds["z"][1] - bounds["z"][0])

        times_norm = times_real / bounds["t"][1]

        n_receivers = len(receiver_coords_norm)
        n_times = len(times_norm)

        receiver_coords_expanded = np.repeat(receiver_coords_norm, n_times, axis=0)
        times_expanded = np.tile(times_norm.reshape(-1, 1), (n_receivers, 1))
        u_obs_flat = u_obs.flatten().reshape(-1, 1)

        coords_t = torch.FloatTensor(receiver_coords_expanded).to(self.device)
        times_t = torch.FloatTensor(times_expanded).to(self.device)
        u_obs_t = torch.FloatTensor(u_obs_flat).to(self.device)

        xs_norm = torch.tensor(init_params_norm[0], device=self.device, requires_grad=True)
        ys_norm = torch.tensor(init_params_norm[1], device=self.device, requires_grad=True)
        zs_norm = torch.tensor(init_params_norm[2], device=self.device, requires_grad=True)
        ts_norm = torch.tensor(init_params_norm[3], device=self.device, requires_grad=True)

        optimizer = optim.Adam([xs_norm, ys_norm, zs_norm, ts_norm], lr=lr)

        best_loss = float('inf')
        best_params_norm = None
        loss_history = []

        pbar = tqdm(range(n_iterations), desc="Local optimization")

        for iteration in pbar:
            optimizer.zero_grad()

            relative_times = times_t - ts_norm
            relative_times = torch.clamp(relative_times, min=0.0)
            inputs = torch.cat([coords_t, relative_times], dim=1)
            u_pred = self.model(inputs)
            loss = torch.mean((u_pred - u_obs_t) ** 2)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params_norm = (xs_norm.item(), ys_norm.item(),
                                    zs_norm.item(), ts_norm.item())

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        pbar.close()

        final_params_real = normalized_to_real_coords(best_params_norm)

        print(f"\n  Local optimization result:")
        print(f"    X = {final_params_real[0]:.1f} m")
        print(f"    Y = {final_params_real[1]:.1f} m")
        print(f"    Z = {final_params_real[2]:.1f} m")
        print(f"    T = {final_params_real[3]:.4f} s")
        print(f"    Loss = {best_loss:.6f}")

        return final_params_real, best_loss, loss_history

    def invert(self, receiver_coords_real, times_real, u_obs):
        print("\n" + "=" * 80)
        print("Source Inversion based on PINN Wavefield Model")
        print(f"Model frequency: {self.freq_hz} Hz")
        print("=" * 80)

        start_time = time.time()

        grid_params, grid_loss = self.grid_search(receiver_coords_real, times_real, u_obs)
        final_params, final_loss, loss_history = self.local_optimization(
            receiver_coords_real, times_real, u_obs, grid_params,
            n_iterations=CONFIG["local_optimization_iterations"],
            lr=CONFIG["local_optimization_lr"]
        )

        convergence_time = time.time() - start_time

        result = {
            'estimated_source': {
                'x': float(final_params[0]),
                'y': float(final_params[1]),
                'z': float(final_params[2]),
                't': float(final_params[3])
            },
            'grid_search_result': {
                'x': float(grid_params[0]),
                'y': float(grid_params[1]),
                'z': float(grid_params[2]),
                't': float(grid_params[3])
            },
            'grid_loss': float(grid_loss),
            'final_loss': float(final_loss),
            'loss_history': loss_history,
            'convergence_time': convergence_time,
            'n_iterations': len(loss_history),
            'frequency_hz': self.freq_hz
        }

        print(f"\nInversion completed!")
        print(f"  Estimated source: ({final_params[0]:.1f}, {final_params[1]:.1f}, {final_params[2]:.1f}) m")
        print(f"  Estimated origin time: {final_params[3]:.4f} s")
        print(f"  Convergence time: {convergence_time:.2f} s")
        print(f"  Final loss: {final_loss:.6f}")

        return result


# 9. Plotting functions
def plot_inversion_result(result, waveform_data, times, receiver_coords, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)

    ax = axes[0, 0]
    ax.semilogy(result['loss_history'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Convergence Curve', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax = axes[0, 1]
    n_show = min(5, waveform_data.shape[0])
    for i in range(n_show):
        trace_norm = waveform_data[i] / (np.max(np.abs(waveform_data[i])) + 1e-10)
        ax.plot(times, trace_norm + i * 2, 'b-', linewidth=0.8)
        ax.axhline(y=i * 2, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Trace (offset)', fontsize=11)
    ax.set_title('Observed Waveforms', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax = axes[1, 0]
    ax.axis('off')
    result_text = f"""
    Source Inversion Results
    =========================
    Estimated Source Location:
      X: {result['estimated_source']['x']:.1f} m
      Y: {result['estimated_source']['y']:.1f} m
      Z: {result['estimated_source']['z']:.1f} m
    Estimated Origin Time:
      T: {result['estimated_source']['t']:.4f} s
    Performance:
      Time: {result['convergence_time']:.2f} s
      Iterations: {result['n_iterations']}
      Final Loss: {result['final_loss']:.6f}
      Grid Loss: {result['grid_loss']:.6f}
    Model:
      Frequency: {result['frequency_hz']} Hz
      Device: {device}
    """
    ax.text(0.1, 0.5, result_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')

    ax = axes[1, 1]
    scatter = ax.scatter(receiver_coords[:, 0], receiver_coords[:, 1],
                         c=receiver_coords[:, 2], cmap='viridis',
                         s=30, alpha=0.7)
    ax.scatter([result['estimated_source']['x']], [result['estimated_source']['y']],
               c='red', marker='*', s=200, label='Estimated Source',
               edgecolors='black', linewidths=1.5)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title('Receivers and Estimated Source', fontsize=12)
    ax.legend(loc='upper right')
    plt.colorbar(scatter, ax=ax, label='Z (m)')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Microseismic Source Inversion Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {save_path}")

    plt.show()


def plot_comparison_with_true(result, true_source=None, save_path=None):
    if true_source is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)

    params = ['X (m)', 'Y (m)', 'Z (m)']
    true_vals = [true_source[0], true_source[1], true_source[2]]
    est_vals = [result['estimated_source']['x'],
                result['estimated_source']['y'],
                result['estimated_source']['z']]

    for i, (param, true_val, est_val) in enumerate(zip(params, true_vals, est_vals)):
        ax = axes[i]
        ax.bar(['True', 'Estimated'], [true_val, est_val],
               color=['green', 'blue'], alpha=0.7)
        ax.set_ylabel(param, fontsize=11)
        ax.set_title(f'{param} Comparison', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        error = abs(est_val - true_val)
        ax.text(0.5, 0.95, f'Error: {error:.1f} m',
                transform=ax.transAxes, ha='center', fontsize=10)

    plt.suptitle('Comparison Between Estimated and True Source', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison saved: {save_path}")

    plt.show()


# 10. Main
def main():
    print("=" * 80)
    print("Section 3.3: Source Inversion based on PINN")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    print("\n[Step 1] Load pretrained wavefield model")
    freq_for_inversion = CONFIG["default_freq"]

    try:
        wavefield_model = load_pretrained_model(freq_for_inversion, MODEL_32_DIR, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Section 3.2 has been run first")
        return

    print("\n[Step 2] Load waveform data")
    waveform_data, times, receiver_coords = load_waveform_data(WAVEFORM_DIR)

    for i in range(waveform_data.shape[0]):
        waveform_data[i] = waveform_data[i] - np.mean(waveform_data[i])

    u_obs_flat = waveform_data.flatten().reshape(-1, 1)
    valid_mask = ~(np.isnan(u_obs_flat) | np.isinf(u_obs_flat))
    u_obs_flat = u_obs_flat[valid_mask.flatten()]

    print(f"  Valid points: {len(u_obs_flat)} / {waveform_data.size}")

    print("\n[Step 3] Perform source inversion")
    inversion = SourceInversion(wavefield_model, device, freq_hz=freq_for_inversion)
    result = inversion.invert(receiver_coords, times, u_obs_flat)

    print("\n[Step 4] Save results")
    json_path = os.path.join(OUTPUT_DIR, 'inversion_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  JSON saved: {json_path}")

    loss_history_path = os.path.join(OUTPUT_DIR, 'loss_history.npy')
    np.save(loss_history_path, np.array(result['loss_history']))
    print(f"  Loss history saved: {loss_history_path}")

    print("\n[Step 5] Generate figures")
    plot_inversion_result(
        result, waveform_data, times, receiver_coords,
        save_path=os.path.join(FIG_DIR, 'inversion_results.png')
    )

    print("\n" + "=" * 60)
    print("Section 3.3 Completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
