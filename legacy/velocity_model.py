
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
SOURCE_EPOCHS = 50
TARGET_EPOCHS = 40
SAMPLE_RATIO = 0.05
BATCH_SIZE = 512

# Project-relative paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(PROJECT_DIR, '预处理', 'data')
OUTPUT_BASE_DIR = os.path.join(PROJECT_DIR, '3.1')

SAVE_DIR = os.path.join(OUTPUT_BASE_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_BASE_DIR, 'figures')
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'model')

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"=" * 80)
print(f"Configuration:")
print(f"  Downsample factor: {DOWNSAMPLE_FACTOR}")
print(f"  Source epochs: {SOURCE_EPOCHS}")
print(f"  Target epochs: {TARGET_EPOCHS}")
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


# 4. Trainer
class Trainer:
    def __init__(self, layers, lr=2e-4):
        self.source = VelocityMLP(layers).to(device)
        self.target = VelocityMLP(layers).to(device)
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.history = {}

    def train_source(self, train_loader, val_loader, epochs=SOURCE_EPOCHS):
        print(f"\nSource domain training (max {epochs} epochs)...")
        start_time = time.time()
        optimizer = optim.Adam(self.source.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        best_loss = 1e9
        wait = 0
        patience = 30
        train_losses, val_losses = [], []

        for epoch in tqdm(range(epochs), desc="Source Training"):
            self.source.train()
            loss_sum = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = self.criterion(self.source(x), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            train_loss = loss_sum / len(train_loader)

            self.source.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += self.criterion(self.source(x), y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                wait = 0
                best_state = self.source.state_dict()
            else:
                wait += 1

            if wait > patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.source.load_state_dict(best_state)
        self.history['source'] = (train_losses, val_losses, best_epoch)
        elapsed = time.time() - start_time
        print(f"Source training done! Best epoch: {best_epoch + 1}, Time: {elapsed:.1f}s")

    def transfer(self):
        self.target.load_state_dict(self.source.state_dict())
        print("Weight transfer completed")

    def train_target(self, train_loader, val_loader, epochs=TARGET_EPOCHS):
        print(f"\nTarget domain fine-tuning (max {epochs} epochs)...")
        start_time = time.time()
        optimizer = optim.Adam(self.target.parameters(), lr=self.lr * 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8)
        best_loss = 1e9
        wait = 0
        patience = 20
        train_losses, val_losses = [], []

        for epoch in tqdm(range(epochs), desc="Target Training"):
            self.target.train()
            loss_sum = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = self.criterion(self.target(x), y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            train_loss = loss_sum / len(train_loader)

            self.target.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += self.criterion(self.target(x), y).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                wait = 0
                best_state = self.target.state_dict()
            else:
                wait += 1

            if wait > patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.target.load_state_dict(best_state)
        self.history['target'] = (train_losses, val_losses, best_epoch)
        elapsed = time.time() - start_time
        print(f"Target training done! Best epoch: {best_epoch + 1}, Time: {elapsed:.1f}s")

    def evaluate(self, test_loader, norm):
        self.target.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = self.target(x).cpu().numpy()
                preds.append(out)
                trues.append(y.numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        preds = norm.denorm(preds)
        trues = norm.denorm(trues)

        rmse = np.sqrt(np.mean((preds - trues) ** 2))
        mae = np.mean(np.abs(preds - trues))
        r2 = 1 - np.sum((preds - trues) ** 2) / np.sum((trues - np.mean(trues)) ** 2)

        print("\n===== Test Metrics =====")
        print(f"RMSE: {rmse:.2f} m/s")
        print(f"MAE: {mae:.2f} m/s")
        print(f"R2: {r2:.4f}")

        np.save(os.path.join(SAVE_DIR, 'predictions.npy'), preds)
        np.save(os.path.join(SAVE_DIR, 'targets.npy'), trues)
        with open(os.path.join(SAVE_DIR, 'metrics.txt'), 'w') as f:
            f.write(f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2: {r2:.4f}")

        return rmse, mae, r2

    def predict_full_velocity_model(self, grid_x, grid_y, grid_z, norm, batch_size=4096):
        print(f"\nPredicting full velocity model: {len(grid_x):,} points...")
        start_time = time.time()
        self.target.eval()
        points = np.stack([grid_x, grid_y, grid_z], axis=1)
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(points), batch_size), desc="Predicting"):
                batch = torch.FloatTensor(points[i:i + batch_size]).to(device)
                pred = self.target(batch).cpu().numpy()
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

        np.save(os.path.join(MODEL_OUTPUT_DIR, 'vp_model.npy'), vp_model)
        metadata = {
            'x_coords': np.unique(grid_x), 'y_coords': np.unique(grid_y), 'z_coords': np.unique(grid_z),
            'dx': np.unique(grid_x)[1] - np.unique(grid_x)[0] if len(np.unique(grid_x)) > 1 else 1,
            'dy': np.unique(grid_y)[1] - np.unique(grid_y)[0] if len(np.unique(grid_y)) > 1 else 1,
            'dz': np.unique(grid_z)[1] - np.unique(grid_z)[0] if len(np.unique(grid_z)) > 1 else 1,
            'shape': vp_model.shape, 'downsample_factor': DOWNSAMPLE_FACTOR
        }
        np.save(os.path.join(MODEL_OUTPUT_DIR, 'metadata.npy'), metadata)
        return vp_model, metadata


# 5. Plotting functions
def plot_training_curves(history, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    titles = ["Source Domain Training", "Target Domain Fine-tuning"]
    for i, key in enumerate(['source', 'target']):
        train, val, best = history[key]
        ax[i].plot(train, linewidth=2, label='Training Loss')
        ax[i].plot(val, linewidth=2, linestyle='--', label='Validation Loss')
        ax[i].scatter(best, val[best], color='red', s=60)
        ax[i].annotate(f'Best Epoch = {best + 1}', (best, val[best]), xytext=(best, val[best] * 1.02),
                       arrowprops=dict(arrowstyle='->'))
        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('MSE')
        ax[i].legend(frameon=False)
        ax[i].grid(False)
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
    axes[0, 1].set_title(f'Predicted ({SAMPLE_RATIO * 100:.0f}% data)')
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
    axes[1, 1].plot(vp_pred[x_mid, ny_mid, :], label='Predicted', linewidth=2, linestyle='--')
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


# 6. Visualization from saved results (no retraining)
def visualize_from_saved():
    print("\n" + "=" * 60)
    print("Loading saved results and generating figures")
    print("=" * 60)

    # Load training history
    history_path = os.path.join(SAVE_DIR, 'training_history.npy')
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_curves(history, save_path=os.path.join(FIG_DIR, 'training_curves.png'))
    else:
        print("Warning: training_history.npy not found")

    # Load velocity models
    vp_pred_path = os.path.join(MODEL_OUTPUT_DIR, 'vp_model.npy')
    vp_true_path = os.path.join(MODEL_OUTPUT_DIR, 'vp_true.npy')
    metadata_path = os.path.join(MODEL_OUTPUT_DIR, 'metadata.npy')

    if os.path.exists(vp_pred_path) and os.path.exists(vp_true_path) and os.path.exists(metadata_path):
        vp_pred = np.load(vp_pred_path)
        vp_true = np.load(vp_true_path)
        metadata = np.load(metadata_path, allow_pickle=True).item()

        # Ensure shape match
        if vp_true.shape != vp_pred.shape:
            min_shape = tuple(min(a, b) for a, b in zip(vp_true.shape, vp_pred.shape))
            vp_true = vp_true[:min_shape[0], :min_shape[1], :min_shape[2]]
            vp_pred = vp_pred[:min_shape[0], :min_shape[1], :min_shape[2]]

        plot_velocity_comparison(vp_true, vp_pred, metadata, save_path=os.path.join(FIG_DIR, 'velocity_comparison.png'))
    else:
        print("Warning: Velocity model files not found")

    # Print metrics
    metrics_path = os.path.join(SAVE_DIR, 'metrics.txt')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            print(f"\nTest metrics:\n{f.read()}")

    print("\nFigures regenerated successfully")


# 7. Main training function
def main():
    print("=" * 80)
    print("Section 3.1: Velocity Model Reconstruction via Transfer Learning")
    print("=" * 80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    train_loader, val_loader, test_loader, norm = load_data(INPUT_DATA_DIR, sample_ratio=SAMPLE_RATIO,
                                                            batch_size=BATCH_SIZE)

    # Train model
    model = Trainer([3, 256, 256, 256, 256, 1])
    model.train_source(train_loader, val_loader)
    model.transfer()
    model.train_target(train_loader, val_loader)

    # Evaluate
    model.evaluate(test_loader, norm)

    # Save training history
    np.save(os.path.join(SAVE_DIR, 'training_history.npy'), model.history)

    # Plot training curves
    plot_training_curves(model.history, save_path=os.path.join(FIG_DIR, 'training_curves.png'))

    # Save model
    torch.save(model.target.state_dict(), os.path.join(MODEL_OUTPUT_DIR, 'final_model.pth'))

    # Generate full velocity model
    print("\n" + "=" * 60)
    print("Generating full velocity model (for Section 3.2)")
    print("=" * 60)

    grid_x, grid_y, grid_z, _, _, _ = create_grid(INPUT_DATA_DIR, downsample_factor=DOWNSAMPLE_FACTOR)
    vp_pred, metadata = model.predict_full_velocity_model(grid_x, grid_y, grid_z, norm)

    # Load and compare with true model
    vp_true = load_true_model(INPUT_DATA_DIR, target_shape=vp_pred.shape)
    if vp_true is not None:
        if vp_true.shape != vp_pred.shape:
            min_shape = tuple(min(a, b) for a, b in zip(vp_true.shape, vp_pred.shape))
            vp_true = vp_true[:min_shape[0], :min_shape[1], :min_shape[2]]
            vp_pred = vp_pred[:min_shape[0], :min_shape[1], :min_shape[2]]
        plot_velocity_comparison(vp_true, vp_pred, metadata, save_path=os.path.join(FIG_DIR, 'velocity_comparison.png'))
        np.save(os.path.join(MODEL_OUTPUT_DIR, 'vp_true.npy'), vp_true)

    print("\n" + "=" * 60)
    print("Section 3.1 completed!")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {SAVE_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(f"Models: {MODEL_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--viz':
        visualize_from_saved()
    else:
        main()
