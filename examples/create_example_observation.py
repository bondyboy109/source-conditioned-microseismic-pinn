"""Create a format-only observation archive for pipeline testing.

The generated zero waveforms are not a scientific validation dataset.
"""

from pathlib import Path

import numpy as np


def main() -> None:
    output = Path(__file__).with_name("format_only_observation.npz")
    n_receiver, n_time = 8, 50
    receiver_coords = np.zeros((n_receiver, 3), dtype=np.float32)
    receiver_coords[:, 0] = np.linspace(0.0, 16000.0, n_receiver)
    receiver_coords[:, 1] = 8000.0
    receiver_coords[:, 2] = 100.0
    times = np.linspace(0.0, 0.98, n_time, dtype=np.float32)
    data = np.zeros((n_receiver, n_time), dtype=np.float32)
    source_truth = np.array([7000.0, 8000.0, 500.0, 0.20], dtype=np.float32)
    np.savez_compressed(
        output,
        data=data,
        times=times,
        receiver_coords=receiver_coords,
        source_truth=source_truth,
    )
    print(f"Wrote format-only archive: {output}")


if __name__ == "__main__":
    main()
