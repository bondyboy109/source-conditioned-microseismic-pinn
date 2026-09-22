import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurable parameters
DOWNSAMPLE_FACTOR = 2
EPOCHS = 60
SAMPLE_RATIO = 0.05
BATCH_SIZE = 512

# Project-relative paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(PROJECT_DIR, '预处理', 'data')
OUTPUT_BASE_DIR = os.path.join(PROJECT_DIR, '4.3 消融实验', '速度模型消融')

SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'figures')
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'model')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"=" * 80)
print(f"Ablation Experiment - No Transfer Learning Configuration:")
print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Sample ratio: {SAMPLE_RATIO}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"=" * 80)

# Plot settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Random seed
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# 1. MLP model
class VelocityMLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Tanh()
        for m in self.layers:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# 2. Normalizer
class Normalizer:
    def fit(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def norm(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def denorm(self, x):
        return x * self.std + self.mean


# 3. Data loading
def load_data(input_data_dir, sample_ratio=SAMPLE_RATIO, batch_size=BATCH_SIZE):
    print("\nLoading data...")
    train_points = np.load(os.path.join(input_data_dir, 'train_points.npy'))
    train_values = np.load(os.path.join(input_data_dir, 'train_values.npy'))
    val_points = np.load(os.path.join(input_data_dir, 'val_points.npy'))
    val_values = np.load(os.path.join(input_data_dir, 'val_values.npy'))
    test_points = np.load(os.path.join(input_data_dir, 'test_points.npy'))
    test_values = np.load(os.path.join(input_data_dir, 'test_values.npy'))

    print(f"Original: train={len(train_points):,}, val={len(val_points):,}, test={len(test_points):,}")

    def sample(x, y):
        n = int(len(x) * sample_ratio)
        idx = np.random.choice(len(x), n, replace=False)
        return x[idx], y[idx]

    train_points, train_values = sample(train_points, train_values)
    val_points, val_values = sample(val_points, val_values)
    test_points, test_values = sample(test_points, test_values)

    print(f"Sampled: train={len(train_points):,}, val={len(val_points):,}, test={len(test_points):,}")

    norm = Normalizer()
    norm.fit(train_values)

    train_values = norm.norm(train_values)
    val_values = norm.norm(val_values)
    test_values = norm.norm(test_values)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_points), torch.FloatTensor(train_values).unsqueeze(1)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_points), torch.FloatTensor(val_values).unsqueeze(1)),
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_points), torch.FloatTensor(test_values).unsqueeze(1)),
                             batch_size=batch_size)

    return train_loader, val_loader, test_loader, norm


# 4. Trainer (no transfer learning version)
class TrainerNoTransfer:
    def __init__(self, layers, lr=2e-4):
        self.model = VelocityMLP(layers).to(device)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.history = {'train_losses': [], 'val_losses': []}
        self.best_epoch = -1

    def train(self, train_loader, val_loader, epochs=EPOCHS):
        print(f"\nTraining (max {epochs} epochs)...")
        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        best_loss = 1e9
        wait = 0
        patience = 30

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            loss_sum = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            train_loss = loss_sum / len(train_loader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += self.criterion(self.model(x), y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_epoch = epoch
                wait = 0
                best_state = self.model.state_dict()
            else:
                wait += 1

            if wait > patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(best_state)
        elapsed = time.time() - start_time
        print(f"Training done! Best epoch: {self.best_epoch + 1}, Time: {elapsed:.1f}s")
        return self.history

    def evaluate(self, test_loader, norm):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = self.model(x).cpu().numpy()
                preds.append(out)
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        preds = norm.denorm(preds)
        trues = norm.denorm(trues)

        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        mae = np.mean(np.abs(preds - trues))
        r2 = 1 - np.sum((preds - trues) ** 2) / np.sum((trues - np.mean(trues)) ** 2)

        print("\n===== Test Metrics (No Transfer Learning) =====")
        print(f"RMSE: {rmse:.2f} m/s")
        print(f"MAE: {mae:.2f} m/s")
        print(f"R2: {r2:.4f}")

        np.save(os.path.join(SAVE_DIR, 'predictions.npy'), preds)
        np.save(os.path.join(SAVE_DIR, 'targets.npy'), trues)
        with open(os.path.join(SAVE_DIR, 'metrics.txt'), 'w') as f:
            f.write(f"Ablation Experiment: No Transfer Learning\n")
            f.write(f"Epochs: {EPOCHS}\n")
            f.write(f"Sample ratio: {SAMPLE_RATIO}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Downsample factor: {DOWNSAMPLE_FACTOR}\n\n")
            f.write(f"RMSE: {rmse:.2f} m/s\n")
            f.write(f"MAE: {mae:.2f} m/s\n")
            f.write(f"R2: {r2:.4f}\n")
            f.write(f"Best epoch: {self.best_epoch + 1}\n")

        return rmse, mae, r2

    def predict_full_velocity_model(self, grid_x, grid_y, grid_z, norm, batch_size=4096):
        print(f"\nPredicting full velocity model: {len(grid_x):,} points...")
        start_time = time.time()
        self.model.eval()
        points = np.stack([grid_x, grid_y, grid_z], axis=1)
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(points), batch_size), desc="Predicting"):
                batch = torch.FloatTensor(points[i:i + batch_size]).to(device)
                pred = self.model(batch).cpu().numpy()
                predictions.append(pred)
        predictions = np.concatenate(predictions, axis=0)
        predictions = norm.denorm(predictions)

        nx = len(np.unique(grid_x))
        ny = len(np.unique(grid_y))
        nz = len(np.unique(grid_z))
        vp_model = predictions.reshape(nx, ny, nz)

        elapsed = time.time() - start_time
        print(f"Prediction done! Shape: {vp_model.shape}, Time: {elapsed:.1f}s")
        print(f"Velocity range: [{vp_model.min():.0f}, {vp_model.max():.0f}] m/s")

        np.save(os.path.join(MODEL_OUTPUT_DIR, 'vp_model_no_transfer.npy'), vp_model)
        metadata = {
            'x_coords': np.unique(grid_x), 'y_coords': np.unique(grid_y), 'z_coords': np.unique(grid_z),
            'dx': np.unique(grid_x)[1] - np.unique(grid_x)[0] if len(np.unique(grid_x)) > 1 else 1,
            'dy': np.unique(grid_y)[1] - np.unique(grid_y)[0] if len(np.unique(grid_y)) > 1 else 1,
            'dz': np.unique(grid_z)[1] - np.unique(grid_z)[0] if len(np.unique(grid_z)) > 1 else 1,
            'shape': vp_model.shape, 'downsample_factor': DOWNSAMPLE_FACTOR
        }
        np.save(os.path.join(MODEL_OUTPUT_DIR, 'metadata_no_transfer.npy'), metadata)
        return vp_model, metadata


# 5. Plotting functions
def plot_training_curve(history, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
    ax.plot(history['train_losses'], linewidth=2, label='Training Loss')
    ax.plot(history['val_losses'], linewidth=2, linestyle='--', label='Validation Loss')

    if history['val_losses']:
        best_epoch = np.argmin(history['val_losses'])
        ax.scatter(best_epoch, history['val_losses'][best_epoch], color='red', s=60, zorder=5)
        ax.annotate(f'Best = {best_epoch + 1}',
                    (best_epoch, history['val_losses'][best_epoch]),
                    xytext=(best_epoch, history['val_losses'][best_epoch] * 1.02),
                    arrowprops=dict(arrowstyle='->'))

    ax.set_title('Training Process (Without Transfer Learning)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(frameon=False)
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_velocity_comparison(vp_true, vp_pred, metadata, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    ny_mid = vp_true.shape[1] // 2

    im1 = axes[0, 0].imshow(vp_true[:, ny_mid, :].T, cmap='jet', aspect='auto',
                            extent=[0, vp_true.shape[0] * metadata['dx'], 0, vp_true.shape[2] * metadata['dz']])
    axes[0, 0].set_title('True Velocity Model')
    axes[0, 0].set_xlabel('X (m)');
    axes[0, 0].set_ylabel('Z (m)')
    plt.colorbar(im1, ax=axes[0, 0], label='Velocity (m/s)')

    im2 = axes[0, 1].imshow(vp_pred[:, ny_mid, :].T, cmap='jet', aspect='auto',
                            extent=[0, vp_pred.shape[0] * metadata['dx'], 0, vp_pred.shape[2] * metadata['dz']])
    axes[0, 1].set_title(f'Predicted (No Transfer, {SAMPLE_RATIO * 100:.0f}% data)')
    axes[0, 1].set_xlabel('X (m)');
    axes[0, 1].set_ylabel('Z (m)')
    plt.colorbar(im2, ax=axes[0, 1], label='Velocity (m/s)')

    error = np.abs(vp_true - vp_pred)
    im3 = axes[1, 0].imshow(error[:, ny_mid, :].T, cmap='hot', aspect='auto',
                            extent=[0, error.shape[0] * metadata['dx'], 0, error.shape[2] * metadata['dz']])
    axes[1, 0].set_title(f'Absolute Error (RMSE: {np.sqrt(np.mean(error ** 2)):.1f} m/s)')
    axes[1, 0].set_xlabel('X (m)');
    axes[1, 0].set_ylabel('Z (m)')
    plt.colorbar(im3, ax=axes[1, 0], label='Error (m/s)')

    x_mid = vp_true.shape[0] // 2
    axes[1, 1].plot(vp_true[x_mid, ny_mid, :], label='True', linewidth=2)
    axes[1, 1].plot(vp_pred[x_mid, ny_mid, :], label='Predicted (No Transfer)', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Velocity Profile at Center')
    axes[1, 1].set_xlabel('Z Index');
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].legend();
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_grid(input_data_dir, downsample_factor=DOWNSAMPLE_FACTOR):
    print("\nCreating grid coordinates...")
    x_coords_full = np.load(os.path.join(input_data_dir, 'x_coords.npy'))
    y_coords_full = np.load(os.path.join(input_data_dir, 'y_coords.npy'))
    z_coords_full = np.load(os.path.join(input_data_dir, 'z_coords.npy'))

    print(f"Full grid: {len(x_coords_full)} x {len(y_coords_full)} x {len(z_coords_full)}")
    x_coords = x_coords_full[::downsample_factor]
    y_coords = y_coords_full[::downsample_factor]
    z_coords = z_coords_full[::downsample_factor]

    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    print(f"Downsampled: {nx} x {ny} x {nz} ({nx * ny * nz:,} points)")

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    return xx.flatten(), yy.flatten(), zz.flatten(), x_coords, y_coords, z_coords


def load_true_model(input_data_dir, target_shape=None):
    vp_path = os.path.join(input_data_dir, 'vp_clean.npy')
    if os.path.exists(vp_path):
        vp_true = np.load(vp_path)
        if target_shape is not None:
            vp_true = vp_true[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
        return vp_true
    return None


# 6. Main
def main():
    print("=" * 80)
    print("Ablation Experiment: Velocity Model Reconstruction Without Transfer Learning")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_loader, val_loader, test_loader, norm = load_data(INPUT_DATA_DIR, sample_ratio=SAMPLE_RATIO,
                                                            batch_size=BATCH_SIZE)

    trainer = TrainerNoTransfer([3, 256, 256, 256, 256, 1])
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)

    rmse, mae, r2 = trainer.evaluate(test_loader, norm)

    print("\n" + "=" * 60)
    print("Ablation Experiment - No Transfer Learning Results:")
    print(f"  RMSE: {rmse:.2f} m/s")
    print(f"  MAE: {mae:.2f} m/s")
    print(f"  R2: {r2:.4f}")
    print("=" * 60)

    plot_training_curve(history, save_path=os.path.join(FIG_DIR, 'training_curve_no_transfer.png'))

    torch.save(trainer.model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, 'model_no_transfer.pth'))

    print("\n" + "=" * 60)
    print("Generating full velocity model")
    print("=" * 60)

    grid_x, grid_y, grid_z, _, _, _ = create_grid(INPUT_DATA_DIR, downsample_factor=DOWNSAMPLE_FACTOR)
    vp_pred, metadata = trainer.predict_full_velocity_model(grid_x, grid_y, grid_z, norm)

    vp_true = load_true_model(INPUT_DATA_DIR, target_shape=vp_pred.shape)
    if vp_true is not None:
        if vp_true.shape != vp_pred.shape:
            min_shape = tuple(min(a, b) for a, b in zip(vp_true.shape, vp_pred.shape))
            vp_true = vp_true[:min_shape[0], :min_shape[1], :min_shape[2]]
            vp_pred = vp_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        plot_velocity_comparison(vp_true, vp_pred, metadata,
                                 save_path=os.path.join(FIG_DIR, 'velocity_comparison_no_transfer.png'))
        np.save(os.path.join(MODEL_OUTPUT_DIR, 'vp_true.npy'), vp_true)

    with open(os.path.join(OUTPUT_BASE_DIR, 'experiment_summary.txt'), 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Ablation Experiment: No Transfer Learning\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Architecture: [3, 256, 256, 256, 256, 1]\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Sample ratio: {SAMPLE_RATIO}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Downsample factor: {DOWNSAMPLE_FACTOR}\n")
        f.write(f"  Learning rate: 2e-4\n\n")
        f.write("Results:\n")
        f.write(f"  RMSE: {rmse:.2f} m/s\n")
        f.write(f"  MAE: {mae:.2f} m/s\n")
        f.write(f"  R2: {r2:.4f}\n")
        f.write(f"  Best epoch: {trainer.best_epoch + 1}\n")

    print("\n" + "=" * 60)
    print("Ablation Experiment Completed!")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Metrics saved to: {SAVE_DIR}/metrics.txt")
    print(f"Summary saved to: {OUTPUT_BASE_DIR}/experiment_summary.txt")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
