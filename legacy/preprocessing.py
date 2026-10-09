import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# Path configuration
data_path = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\overthrust.vites'

OUTPUT_BASE_DIR = r'E:\04 本科毕业设计\007 本科毕业设计2025-2026级\202283710053_王馨仪\10_202283710053_王馨仪毕业设计代码\预处理'
DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print(f"Output dir: {OUTPUT_BASE_DIR}")
print(f"Data dir: {DATA_DIR}")
print(f"Figure dir: {FIG_DIR}")


# Font settings
def setup_english_font():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    print("Using English font")


# 1. Read velocity model
def read_velocity_model(data_path, nx=801, ny=801, nz=187):
    print("Reading velocity model...")
    data = np.fromfile(data_path, dtype='>f4')
    print(f"Read {len(data):,} data points")
    vp = data.reshape(nx, ny, nz, order='F')
    print(f"Shape: {vp.shape}")
    print(f"Velocity range: [{vp.min():.0f}, {vp.max():.0f}] m/s")
    return vp


# 2. Generate coordinate grid
def generate_coordinates(vp, dx=20.0, dy=20.0, dz=20.0):
    print("\nGenerating coordinate grid...")
    nx, ny, nz = vp.shape
    x_coords = np.arange(nx) * dx
    y_coords = np.arange(ny) * dy
    z_coords = np.arange(nz) * dz
    print(f"X: [{x_coords[0]:.0f}, {x_coords[-1]:.0f}] m, {nx} points")
    print(f"Y: [{y_coords[0]:.0f}, {y_coords[-1]:.0f}] m, {ny} points")
    print(f"Z: [{z_coords[0]:.0f}, {z_coords[-1]:.0f}] m, {nz} points")
    return x_coords, y_coords, z_coords


# 3. Create train/val/test dataset
def create_dataset(vp, x_coords, y_coords, z_coords,
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                   random_seed=42):
    print("\nCreating dataset...")
    nx, ny, nz = vp.shape

    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    values = vp.flatten()
    total_samples = len(points)
    print(f"Total samples: {total_samples:,}")

    np.random.seed(random_seed)
    indices = np.random.permutation(total_samples)

    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_points = points[train_idx]
    train_values = values[train_idx]
    val_points = points[val_idx]
    val_values = values[val_idx]
    test_points = points[test_idx]
    test_values = values[test_idx]

    print(f"Train: {len(train_points):,} ({len(train_points) / total_samples * 100:.1f}%)")
    print(f"Val: {len(val_points):,} ({len(val_points) / total_samples * 100:.1f}%)")
    print(f"Test: {len(test_points):,} ({len(test_points) / total_samples * 100:.1f}%)")

    np.save(os.path.join(DATA_DIR, 'train_points.npy'), train_points)
    np.save(os.path.join(DATA_DIR, 'train_values.npy'), train_values)
    np.save(os.path.join(DATA_DIR, 'val_points.npy'), val_points)
    np.save(os.path.join(DATA_DIR, 'val_values.npy'), val_values)
    np.save(os.path.join(DATA_DIR, 'test_points.npy'), test_points)
    np.save(os.path.join(DATA_DIR, 'test_values.npy'), test_values)
    np.save(os.path.join(DATA_DIR, 'vp_clean.npy'), vp)
    np.save(os.path.join(DATA_DIR, 'x_coords.npy'), x_coords)
    np.save(os.path.join(DATA_DIR, 'y_coords.npy'), y_coords)
    np.save(os.path.join(DATA_DIR, 'z_coords.npy'), z_coords)

    print(f"Data saved to: {DATA_DIR}")
    return train_points, train_values, val_points, val_values, test_points, test_values


# 4. 3D voxel visualization
def visualize_3d_voxel(vp, subsample=10, save_path=None):
    print("\nGenerating 3D voxel visualization...")
    nx, ny, nz = vp.shape

    x_indices = np.arange(0, nx, subsample)
    y_indices = np.arange(0, ny, subsample)
    z_indices = np.arange(0, nz, subsample)

    X, Y, Z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    vp_sampled = vp[x_indices][:, y_indices][:, :, z_indices]

    points_x = X.flatten() * 20.0 / 1000
    points_y = Y.flatten() * 20.0 / 1000
    points_z = Z.flatten() * 20.0 / 1000
    values = vp_sampled.flatten()

    print(f"Sampled points: {len(points_x):,}")

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(points_x, points_y, points_z,
                         c=values, cmap='jet', s=1, alpha=0.5,
                         norm=Normalize(vmin=2000, vmax=6000))

    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title('3D Voxel Visualization of Overthrust Velocity Model', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Velocity (m/s)', fontsize=10)

    ax.view_init(elev=25, azim=45)
    ax.set_xlim([0, nx * 20 / 1000])
    ax.set_ylim([0, ny * 20 / 1000])
    ax.set_zlim([0, nz * 20 / 1000])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# 5. 2D slices visualization
def visualize_2d_slices(vp, save_path=None):
    print("\nGenerating 2D slices visualization...")
    nx, ny, nz = vp.shape

    slice_x = nx // 2
    slice_y = ny // 2
    slice_z = nz // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    x_coords = np.arange(nx) * 20.0 / 1000
    y_coords = np.arange(ny) * 20.0 / 1000
    z_coords = np.arange(nz) * 20.0 / 1000

    # X slice
    ax1 = axes[0, 0]
    slice_x_data = vp[slice_x, :, :].T
    im1 = ax1.imshow(slice_x_data, origin='lower', aspect='auto',
                     extent=[0, ny * 20 / 1000, 0, nz * 20 / 1000],
                     cmap='jet', vmin=vp.min(), vmax=vp.max())
    ax1.set_xlabel('Y (km)')
    ax1.set_ylabel('Z (km)')
    ax1.set_title(f'X Slice (x={slice_x * 20 / 1000:.1f} km)')
    plt.colorbar(im1, ax=ax1, label='Velocity (m/s)')

    # Y slice
    ax2 = axes[0, 1]
    slice_y_data = vp[:, slice_y, :].T
    im2 = ax2.imshow(slice_y_data, origin='lower', aspect='auto',
                     extent=[0, nx * 20 / 1000, 0, nz * 20 / 1000],
                     cmap='jet', vmin=vp.min(), vmax=vp.max())
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Z (km)')
    ax2.set_title(f'Y Slice (y={slice_y * 20 / 1000:.1f} km)')
    plt.colorbar(im2, ax=ax2, label='Velocity (m/s)')

    # Z slice
    ax3 = axes[1, 0]
    slice_z_data = vp[:, :, slice_z].T
    im3 = ax3.imshow(slice_z_data, origin='lower', aspect='auto',
                     extent=[0, nx * 20 / 1000, 0, ny * 20 / 1000],
                     cmap='jet', vmin=vp.min(), vmax=vp.max())
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Y (km)')
    ax3.set_title(f'Z Slice (z={slice_z * 20 / 1000:.1f} km)')
    plt.colorbar(im3, ax=ax3, label='Velocity (m/s)')

    # Velocity vs depth
    ax4 = axes[1, 1]
    depth_avg = np.mean(vp, axis=(0, 1))
    depths = z_coords
    ax4.plot(depth_avg, depths, 'b-', linewidth=2)
    ax4.fill_betweenx(depths, depth_avg - vp.std(axis=(0, 1)),
                      depth_avg + vp.std(axis=(0, 1)), alpha=0.3, color='blue')
    ax4.set_xlabel('Average Velocity (m/s)')
    ax4.set_ylabel('Depth (km)')
    ax4.set_title('Average Velocity vs Depth')
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    plt.suptitle('Slices of Overthrust Velocity Model', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# Main
if __name__ == "__main__":
    print("=" * 80)
    print("Overthrust Velocity Model Preprocessing")
    print("=" * 80)

    setup_english_font()

    vp = read_velocity_model(data_path)
    x_coords, y_coords, z_coords = generate_coordinates(vp, dx=20.0, dy=20.0, dz=20.0)

    create_dataset(vp, x_coords, y_coords, z_coords,
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    visualize_3d_voxel(vp, subsample=10, save_path=os.path.join(FIG_DIR, '3d_voxel.png'))
    visualize_2d_slices(vp, save_path=os.path.join(FIG_DIR, '2d_slices.png'))

    print("\n" + "=" * 60)
    print("All done!")
    print(f"Data: {DATA_DIR}")
    print(f"Figures: {FIG_DIR}")
    print("=" * 60)