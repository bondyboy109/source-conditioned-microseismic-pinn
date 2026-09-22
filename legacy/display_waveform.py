import os
import numpy as np
import matplotlib.pyplot as plt

# Try to import obspy
try:
    from obspy import read

    USE_OBSPY = True
    print("Loaded ObsPy library")
except ImportError:
    USE_OBSPY = False
    print("ObsPy not installed, trying segyio")

    try:
        import segyio

        USE_SEGYIO = True
        print("Loaded segyio library")
    except ImportError:
        USE_SEGYIO = False
        print("Please install: pip install obspy")

# Project-relative paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, '3.3波形数据')
SGY_FILE = os.path.join(OUTPUT_DIR, 'npr3_gathers.sgy')
FIG_DIR = os.path.join(PROJECT_DIR, '图片')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# Method 1: Read with ObsPy
def read_with_obspy(filepath):
    print("\nReading with ObsPy...")
    try:
        stream = read(filepath, headonly=False)
        print(f"  Contains {len(stream)} traces")

        n_traces = len(stream)
        n_samples = stream[0].stats.npts if n_traces > 0 else 0
        print(f"  Samples: {n_samples}")
        print(f"  Sampling rate: {1.0 / stream[0].stats.delta:.1f} Hz")

        max_traces = min(n_traces, 1000)
        data = np.zeros((max_traces, n_samples))
        for i in range(max_traces):
            data[i] = stream[i].data

        times = np.arange(n_samples) * stream[0].stats.delta

        return {
            'data': data,
            'times': times,
            'sampling_rate': 1.0 / stream[0].stats.delta,
            'npts': n_samples,
            'n_traces_total': n_traces,
            'n_traces_loaded': max_traces,
            'data_type': 'obspy'
        }
    except Exception as e:
        print(f"  ObsPy failed: {e}")
        return None


# Method 2: Read with segyio
def read_with_segyio(filepath):
    print("\nReading with segyio...")
    try:
        with segyio.open(filepath, "r", ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = f.samples.size
            print(f"  Total traces: {n_traces}")
            print(f"  Samples per trace: {n_samples}")

            max_traces = min(n_traces, 500)
            print(f"  Loading {max_traces} traces")

            data = np.zeros((max_traces, n_samples), dtype=np.float32)
            for i in range(max_traces):
                trace_data = f.trace[i]
                if np.any(np.isnan(trace_data)) or np.any(np.isinf(trace_data)):
                    trace_data = np.zeros_like(trace_data)
                data[i] = trace_data

            dt = 0.004
            try:
                if hasattr(f, 'bin'):
                    dt = f.bin.hdr[segyio.TraceField.SampleInterval] / 1000000.0
            except:
                dt = 0.004

            times = np.arange(n_samples) * dt

            return {
                'data': data,
                'times': times,
                'n_traces_total': n_traces,
                'n_traces_loaded': max_traces,
                'n_samples': n_samples,
                'dt': dt,
                'sampling_rate': 1.0 / dt,
                'data_type': 'segyio'
            }
    except Exception as e:
        print(f"  segyio failed: {e}")
        return None


# Method 3: Simple numpy read
def read_sgy_simple(filepath, max_traces=500):
    print("\nTrying simple read...")
    try:
        file_size = os.path.getsize(filepath)
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")

        n_samples = 2049
        dt = 0.004
        bytes_per_sample = 4
        bytes_per_trace = n_samples * bytes_per_sample + 240

        estimated_traces = (file_size - 3600) // bytes_per_trace
        print(f"  Estimated traces: {estimated_traces}")

        n_traces = min(estimated_traces, max_traces)

        with open(filepath, 'rb') as f:
            f.seek(3600 + 240)
            data = np.zeros((n_traces, n_samples), dtype=np.float32)

            for i in range(n_traces):
                trace_data = np.fromfile(f, dtype=np.float32, count=n_samples)
                if len(trace_data) == n_samples:
                    trace_data = np.nan_to_num(trace_data, nan=0.0, posinf=0.0, neginf=0.0)
                    trace_data = np.clip(trace_data, -1e6, 1e6)
                    data[i] = trace_data
                else:
                    break

                if i < n_traces - 1:
                    f.seek(240, 1)

            valid_traces = ~np.all(data == 0, axis=1)
            data = data[valid_traces]
            print(f"  Loaded {len(data)} valid traces")

            times = np.arange(n_samples) * dt

            return {
                'data': data,
                'times': times,
                'n_traces_total': estimated_traces,
                'n_traces_loaded': len(data),
                'n_samples': n_samples,
                'dt': dt,
                'sampling_rate': 1.0 / dt,
                'data_type': 'simple'
            }
    except Exception as e:
        print(f"  Simple read failed: {e}")
        return None


# Data cleaning
def clean_data(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    for i in range(data.shape[0]):
        mean_val = np.mean(data[i])
        std_val = np.std(data[i])
        if std_val > 0:
            threshold = 3 * std_val
            data[i] = np.clip(data[i], mean_val - threshold, mean_val + threshold)
    return data


# Save as NumPy
def save_as_numpy(data_info, output_dir, base_filename='processed_waveform_data'):
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)

    npz_path = os.path.join(output_dir, f'{base_filename}.npz')
    np.savez_compressed(
        npz_path,
        data=data_info['data'],
        times=data_info['times'],
        sampling_rate=data_info.get('sampling_rate', 250),
        dt=data_info.get('dt', 0.004),
        n_traces_total=data_info.get('n_traces_total', data_info['data'].shape[0]),
        n_traces_loaded=data_info.get('n_traces_loaded', data_info['data'].shape[0]),
        n_samples=data_info['data'].shape[1],
        data_type=data_info.get('data_type', 'unknown')
    )
    print(f"Saved compressed data: {npz_path}")

    npy_path = os.path.join(output_dir, f'{base_filename}_data.npy')
    np.save(npy_path, data_info['data'])
    print(f"Saved data array: {npy_path}")

    times_path = os.path.join(output_dir, f'{base_filename}_times.npy')
    np.save(times_path, data_info['times'])
    print(f"Saved time axis: {times_path}")

    metadata_path = os.path.join(output_dir, f'{base_filename}_metadata.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Waveform Data Metadata\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source file: {SGY_FILE}\n")
        f.write(f"Data shape: {data_info['data'].shape}\n")
        f.write(f"Time range: [{data_info['times'][0]:.4f}, {data_info['times'][-1]:.4f}] s\n")
        f.write(f"Sampling interval: {data_info['times'][1] - data_info['times'][0]:.4f} s\n")
        f.write(f"Sampling rate: {data_info.get('sampling_rate', 250):.2f} Hz\n")
        f.write(f"Total traces: {data_info.get('n_traces_total', 'Unknown')}\n")
        f.write(f"Loaded traces: {data_info.get('n_traces_loaded', data_info['data'].shape[0])}\n")
        f.write(f"Data range: [{np.min(data_info['data']):.6f}, {np.max(data_info['data']):.6f}]\n")
        f.write(f"Data mean: {np.mean(data_info['data']):.6f}\n")
        f.write(f"Data std: {np.std(data_info['data']):.6f}\n")
        f.write(f"Data type: {data_info.get('data_type', 'unknown')}\n")
    print(f"Saved metadata: {metadata_path}")

    return npz_path


# Plotting functions
def plot_waveform_trace(data, times, title="Waveform", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(times, data, 'b-', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_wiggle_section(data, times, title="Wiggle Plot", save_path=None):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    n_traces = min(data.shape[0], 200)
    data_subset = data[:n_traces]
    vmax = np.percentile(np.abs(data_subset), 95)

    im = ax.imshow(data_subset.T, cmap='seismic', aspect='auto',
                   extent=[0, n_traces - 1, times[-1], times[0]],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_seismic_section(data, times, title="Seismic Section", save_path=None):
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    n_traces = min(data.shape[0], 100)
    data_norm = data[:n_traces].copy()

    for i in range(n_traces):
        max_val = np.max(np.abs(data_norm[i]))
        if max_val > 0 and max_val != np.inf:
            data_norm[i] = data_norm[i] / max_val * 0.8

    for i in range(n_traces):
        trace = data_norm[i]
        x_offset = i
        ax.fill_betweenx(times, x_offset, x_offset + trace,
                         where=trace > 0, color='black', alpha=0.5, linewidth=0)
        ax.fill_betweenx(times, x_offset, x_offset + trace,
                         where=trace < 0, color='red', alpha=0.5, linewidth=0)
        ax.plot(x_offset + trace, times, 'k-', linewidth=0.8)

    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, n_traces - 0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_spectrum(data, sampling_rate, title="Frequency Spectrum", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    if data.ndim > 1:
        data_flat = data.flatten()
    else:
        data_flat = data

    data_flat = data_flat[np.isfinite(data_flat)]

    if len(data_flat) > 0:
        n = len(data_flat)
        freq = np.fft.rfftfreq(n, 1 / sampling_rate)
        spectrum = np.abs(np.fft.rfft(data_flat))
        spectrum = spectrum / (np.max(spectrum) + 1e-10)
        ax.plot(freq, 20 * np.log10(spectrum + 1e-10), 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Magnitude (dB)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, min(100, freq[-1]))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def verify_loading(npz_path):
    print("\n" + "=" * 60)
    print("Verifying data loading...")
    print("=" * 60)
    loaded = np.load(npz_path)
    print(f"Successfully loaded: {npz_path}")
    print(f"  Keys: {list(loaded.keys())}")
    print(f"  Data shape: {loaded['data'].shape}")
    print(f"  Time range: [{loaded['times'][0]:.4f}, {loaded['times'][-1]:.4f}] s")
    return loaded


# Main
def main():
    print("=" * 80)
    print("Read, Process, and Save SGY Waveform Data")
    print("=" * 80)

    if not os.path.exists(SGY_FILE):
        print(f"\nError: File not found - {SGY_FILE}")
        return

    print(f"\nSource file: {SGY_FILE}")
    print(f"File size: {os.path.getsize(SGY_FILE) / 1024 / 1024:.2f} MB")

    data_info = None

    if USE_OBSPY:
        data_info = read_with_obspy(SGY_FILE)

    if data_info is None and USE_SEGYIO:
        data_info = read_with_segyio(SGY_FILE)

    if data_info is None:
        data_info = read_sgy_simple(SGY_FILE, max_traces=500)

    if data_info is None:
        print("\nAll reading methods failed")
        print("Please install: pip install obspy")
        return

    print("\nCleaning data...")
    data_info['data'] = clean_data(data_info['data'])

    print("\n" + "=" * 50)
    print("Data Information:")
    print("=" * 50)
    print(f"  Shape: {data_info['data'].shape}")
    print(f"  Time range: [{data_info['times'][0]:.4f}, {data_info['times'][-1]:.4f}] s")
    print(f"  Data range: [{np.min(data_info['data']):.4f}, {np.max(data_info['data']):.4f}]")
    print(f"  Mean: {np.mean(data_info['data']):.4f}")
    print(f"  Std: {np.std(data_info['data']):.4f}")

    save_path = save_as_numpy(data_info, OUTPUT_DIR, 'npr3_gathers_processed')
    verify_loading(save_path)

    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    data = data_info['data']
    times = data_info['times']

    if data.shape[0] > 100:
        data_for_plot = data[:100]
        print(f"  Using first 100 traces for plotting (total {data.shape[0]})")
    else:
        data_for_plot = data

    plot_wiggle_section(data_for_plot, times,
                        title=f"Seismic Section - {os.path.basename(SGY_FILE)}",
                        save_path=os.path.join(FIG_DIR, 'seismic_section_density.png'))

    plot_seismic_section(data_for_plot, times,
                         title="Seismic Section (Wiggle)",
                         save_path=os.path.join(FIG_DIR, 'seismic_section_wiggle.png'))

    if 'sampling_rate' in data_info:
        plot_spectrum(data_for_plot, data_info['sampling_rate'],
                      title="Frequency Spectrum",
                      save_path=os.path.join(FIG_DIR, 'spectrum.png'))

    plot_waveform_trace(data[0], times,
                        title="Trace 0 Waveform",
                        save_path=os.path.join(FIG_DIR, 'trace_0.png'))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print(f"  - npr3_gathers_processed.npz")
    print(f"  - npr3_gathers_processed_data.npy")
    print(f"  - npr3_gathers_processed_times.npy")
    print(f"  - npr3_gathers_processed_metadata.txt")
    print(f"\nFigures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
