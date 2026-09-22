
import os
import numpy as np
import matplotlib.pyplot as plt


# 1. Font settings
def setup_english_font():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    print("Using English font")


# 2. Read velocity model
def read_velocity_model(data_path, nx=801, ny=801, nz=187):
    print("Reading velocity model...")
    data = np.fromfile(data_path, dtype='>f4')
    vp = data.reshape(nx, ny, nz, order='F')
    print(f"Shape: {vp.shape}")
    print(f"Velocity range: {vp.min():.0f} ~ {vp.max():.0f} m/s")
    return vp


# 3. Visualize mean over X and mean over Y
def visualize_mean_xy(vp, save_path=None):
    print("\nGenerating mean heatmaps (X & Y)...")
    nx, ny, nz = vp.shape

    # Mean calculations
    vp_x_mean = np.mean(vp, axis=0)  # (ny, nz) -> V(y,z)
    vp_y_mean = np.mean(vp, axis=1)  # (nx, nz) -> V(x,z)

    # Transpose (z downward)
    vp_x_mean = vp_x_mean.T  # (nz, ny)
    vp_y_mean = vp_y_mean.T  # (nz, nx)

    # Coordinates (convert to km)
    x = np.arange(nx) * 20.0 / 1000
    y = np.arange(ny) * 20.0 / 1000
    z = np.arange(nz) * 20.0 / 1000

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmin, vmax = vp.min(), vp.max()

    # Left: mean over X -> V(y,z)
    im1 = axes[0].imshow(
        vp_x_mean,
        extent=[y.min(), y.max(), z.min(), z.max()],
        aspect='auto',
        cmap='jet',
        vmin=vmin, vmax=vmax
    )
    axes[0].invert_yaxis()
    axes[0].set_title('Mean over X -> V(y,z)')
    axes[0].set_xlabel('Y (km)')
    axes[0].set_ylabel('Depth (km)')

    # Right: mean over Y -> V(x,z)
    im2 = axes[1].imshow(
        vp_y_mean,
        extent=[x.min(), x.max(), z.min(), z.max()],
        aspect='auto',
        cmap='jet',
        vmin=vmin, vmax=vmax
    )
    axes[1].invert_yaxis()
    axes[1].set_title('Mean over Y -> V(x,z)')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Depth (km)')

    # Colorbar
    cbar = fig.colorbar(im1, ax=axes, location='right', shrink=0.85)
    cbar.set_label('Velocity (m/s)')

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()


# Main
if __name__ == "__main__":
    data_path = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\overthrust.vites'

    output_dir = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\figures'
    os.makedirs(output_dir, exist_ok=True)

    setup_english_font()

    vp = read_velocity_model(data_path)

    visualize_mean_xy(
        vp,
        save_path=os.path.join(output_dir, '11_mean_x_vs_mean_y.png')
    )

    print("\nDone")