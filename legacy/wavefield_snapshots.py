import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

# Paths
VELOCITY_MODEL_DIR = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\velocity_model'
FIG_DIR = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\3.2的figure_en'
os.makedirs(FIG_DIR, exist_ok=True)

# Plot settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

print("Using CPU for simulation")


# 1. Load velocity model
def load_velocity_model():
    vp_path = os.path.join(VELOCITY_MODEL_DIR, 'vp_model.npy')
    metadata_path = os.path.join(VELOCITY_MODEL_DIR, 'metadata.npy')

    if not os.path.exists(vp_path):
        raise FileNotFoundError(f"Please run 3.1 first: {vp_path}")

    vp_model = np.load(vp_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()

    print(f"\nLoaded velocity model:")
    print(f"  Shape: {vp_model.shape}")
    print(f"  Velocity range: [{vp_model.min():.0f}, {vp_model.max():.0f}] m/s")

    return vp_model, metadata


# 2. 2D finite-difference wave simulator
class WaveSimulator:
    def __init__(self, vp_2d, dx, dz, dt=None):
        self.vp = vp_2d
        self.nx, self.nz = vp_2d.shape
        self.dx = dx
        self.dz = dz

        vp_max = np.max(vp_2d)
        if dt is None:
            self.dt = 0.8 / (vp_max * np.sqrt(1 / dx ** 2 + 1 / dz ** 2))
        else:
            self.dt = dt

        print(f"  Grid: {self.nx} x {self.nz}")
        print(f"  dt = {self.dt:.6f}s")
        cfl = vp_max * self.dt * np.sqrt(1 / dx ** 2 + 1 / dz ** 2)
        print(f"  CFL: {cfl:.3f} ({'stable' if cfl < 1 else 'unstable'})")

    def ricker_wavelet(self, t, f0):
        tau = np.pi * f0 * (t - 1.0 / f0)
        return (1 - 2 * tau ** 2) * np.exp(-tau ** 2)

    def simulate(self, source_x, source_z, frequency, n_time_steps=500):
        print(f"\n  Simulating {frequency}Hz source...")

        u_prev = np.zeros((self.nx, self.nz))
        u_curr = np.zeros((self.nx, self.nz))

        times = np.arange(0, n_time_steps * self.dt, self.dt)
        source_wavelet = self.ricker_wavelet(times, frequency)

        n_snapshots = 12
        snapshot_interval = max(1, n_time_steps // n_snapshots)
        snapshots = []
        snapshot_times = []

        boundary_width = 20
        pml = self._create_pml(boundary_width)

        for t_step in range(n_time_steps):
            laplacian = np.zeros_like(u_curr)
            laplacian[1:-1, 1:-1] = (
                    (u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / self.dx ** 2 +
                    (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / self.dz ** 2
            )

            u_next = 2 * u_curr - u_prev + (self.dt ** 2) * (self.vp ** 2) * laplacian

            if t_step < len(source_wavelet):
                u_next[source_x, source_z] += source_wavelet[t_step]

            u_next = u_next * pml

            if t_step % snapshot_interval == 0 and len(snapshots) < n_snapshots:
                snapshots.append(u_next.copy())
                snapshot_times.append(t_step * self.dt)

            u_prev, u_curr = u_curr, u_next

        print(f"  Generated {len(snapshots)} snapshots")
        return snapshots, snapshot_times

    def _create_pml(self, width):
        pml = np.ones((self.nx, self.nz))
        for i in range(width):
            factor = np.exp(-0.05 * (width - i) ** 2 / width ** 2)
            pml[i, :] *= factor
            pml[-i - 1, :] *= factor
            pml[:, i] *= factor
            pml[:, -i - 1] *= factor
        return pml


# 3. Plotting functions
def plot_wavefield_snapshots(snapshots, times, vp_2d, dx, dz, freq,
                             source_x=None, source_z=None, save_path=None):
    n_snapshots = len(snapshots)
    n_rows, n_cols = 3, 4

    if n_snapshots < n_rows * n_cols:
        snapshots_padded = snapshots + [None] * (n_rows * n_cols - n_snapshots)
        times_padded = times + [None] * (n_rows * n_cols - n_snapshots)
    else:
        snapshots_padded = snapshots[:n_rows * n_cols]
        times_padded = times[:n_rows * n_cols]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), dpi=150)
    axes = axes.flatten()

    vp_bg = vp_2d.T
    x_max = vp_2d.shape[0] * dx
    z_max = vp_2d.shape[1] * dz

    source_x_m = source_x * dx if source_x is not None else None
    source_z_m = source_z * dz if source_z is not None else None

    all_vals = np.concatenate([s.flatten() for s in snapshots if s is not None])
    vmax = np.percentile(np.abs(all_vals), 95)
    vmin = -vmax

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    sm.set_array([])

    for idx in range(len(snapshots_padded)):
        ax = axes[idx]

        if snapshots_padded[idx] is not None:
            ax.imshow(vp_bg, cmap='gray', aspect='auto', alpha=0.4,
                      extent=[0, x_max, 0, z_max], origin='lower')

            ax.imshow(snapshots_padded[idx].T, cmap='seismic', alpha=0.85, aspect='auto',
                      vmin=vmin, vmax=vmax, extent=[0, x_max, 0, z_max], origin='lower')

            if source_x_m is not None:
                ax.scatter(source_x_m, source_z_m, marker='*', s=200,
                           c='white', edgecolors='black', linewidths=1.5, zorder=10)

            if idx >= 8:
                ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Z (m)', fontsize=11)
            ax.set_title(f't = {times_padded[idx]:.3f} s', fontsize=11)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, z_max)

            xticks = np.arange(0, x_max + 1, x_max // 3)
            yticks = np.arange(0, z_max + 1, z_max // 3)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.tick_params(labelsize=9)
        else:
            ax.axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Amplitude', fontsize=12)

    plt.suptitle(f'Wavefield Propagation - {freq} Hz Source', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_velocity_with_source(vp_2d, dx, dz, source_x=None, source_z=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    x_max = vp_2d.shape[0] * dx
    z_max = vp_2d.shape[1] * dz

    im = ax.imshow(vp_2d.T, cmap='jet', aspect='auto',
                   extent=[0, x_max, 0, z_max], origin='lower')

    if source_x is not None and source_z is not None:
        source_x_m = source_x * dx
        source_z_m = source_z * dz
        ax.scatter(source_x_m, source_z_m, marker='*', s=300,
                   c='white', edgecolors='black', linewidths=2, zorder=10, label='Source')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Velocity Model with Source Location', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_frequency_comparison(all_snapshots, all_times, vp_2d, dx, dz,
                              source_x=None, source_z=None, save_path=None):
    freqs = [5, 10, 20, 40, 60]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    axes = axes.flatten()

    vp_bg = vp_2d.T
    x_max = vp_2d.shape[0] * dx
    z_max = vp_2d.shape[1] * dz
    source_x_m = source_x * dx if source_x is not None else None
    source_z_m = source_z * dz if source_z is not None else None

    all_freq_vals = []
    for freq in freqs:
        if freq in all_snapshots and all_snapshots[freq]:
            all_freq_vals.extend([s.flatten() for s in all_snapshots[freq]])
    if all_freq_vals:
        all_vals_flat = np.concatenate(all_freq_vals)
        vmax = np.percentile(np.abs(all_vals_flat), 95)
    else:
        vmax = 1.0
    vmin = -vmax

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    sm.set_array([])

    for idx, freq in enumerate(freqs):
        ax = axes[idx]
        ax.imshow(vp_bg, cmap='gray', aspect='auto', alpha=0.4,
                  extent=[0, x_max, 0, z_max], origin='lower')

        if freq in all_snapshots and all_snapshots[freq]:
            ax.imshow(all_snapshots[freq][-1].T, cmap='seismic', alpha=0.85, aspect='auto',
                      vmin=vmin, vmax=vmax, extent=[0, x_max, 0, z_max], origin='lower')

        if source_x_m is not None:
            ax.scatter(source_x_m, source_z_m, marker='*', s=200,
                       c='white', edgecolors='black', linewidths=1.5, zorder=10)

        ax.set_title(f'{freq} Hz', fontsize=13, fontweight='bold')
        if idx >= 3:
            ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Z (m)', fontsize=11)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, z_max)

    axes[5].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Amplitude', fontsize=12)

    plt.suptitle('Wavefield Comparison at Different Frequencies', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# 4. Main
def main():
    print("=" * 80)
    print("Section 3.2: Wavefield Propagation Snapshots")
    print("=" * 80)

    vp_model, metadata = load_velocity_model()

    ny_mid = vp_model.shape[1] // 2
    vp_2d = vp_model[:, ny_mid, :]
    print(f"\nExtracted 2D profile (Y={ny_mid}): {vp_2d.shape}")

    dx = metadata['dx']
    dz = metadata['dz']

    source_x = vp_2d.shape[0] // 2
    source_z = vp_2d.shape[1] // 6

    print(f"\nSimulation parameters:")
    print(f"  Grid: {vp_2d.shape[0]} x {vp_2d.shape[1]}")
    print(f"  dx={dx}m, dz={dz}m")
    print(f"  Source: ({source_x * dx:.0f}m, {source_z * dz:.0f}m)")

    plot_velocity_with_source(vp_2d, dx, dz, source_x, source_z,
                              save_path=os.path.join(FIG_DIR, 'velocity_with_source.png'))

    simulator = WaveSimulator(vp_2d, dx, dz)
    FREQUENCIES = [5, 10, 20, 40, 60]
    all_snapshots = {}
    all_times = {}

    for freq in FREQUENCIES:
        print(f"\n{'=' * 50}\nFrequency: {freq} Hz\n{'=' * 50}")
        snapshots, times = simulator.simulate(source_x, source_z, freq, n_time_steps=500)
        all_snapshots[freq] = snapshots
        all_times[freq] = times

        plot_wavefield_snapshots(
            snapshots, times, vp_2d, dx, dz, freq,
            source_x=source_x, source_z=source_z,
            save_path=os.path.join(FIG_DIR, f'wavefield_{freq}Hz.png')
        )

    plot_frequency_comparison(all_snapshots, all_times, vp_2d, dx, dz,
                              source_x=source_x, source_z=source_z,
                              save_path=os.path.join(FIG_DIR, 'frequency_comparison.png'))

    print("\n" + "=" * 60)
    print("Section 3.2 Completed!")
    print(f"Images saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()