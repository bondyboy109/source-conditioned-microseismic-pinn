"""Draw exported FD and network receiver traces using Matplotlib."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    with np.load(args.traces) as data:
        receiver = data["receiver_m"]
        for column, name in enumerate(("train", "test")):
            t = data[f"{name}_times_s"]
            fd = data[f"{name}_fd"] * 1e6
            prediction = data[f"{name}_predicted"] * 1e6
            numerator = np.sum((fd - prediction) ** 2)
            denominator = np.sum(fd ** 2)
            l2 = np.sqrt(numerator / denominator)
            axes[0, column].plot(t, fd, label="FD", linewidth=1.3)
            axes[0, column].plot(t, prediction, label="Receiver-trained model",
                                 linewidth=1.0)
            axes[0, column].set_title(f"{name.title()} source, receiver {receiver.astype(int)} m; "
                                      f"trace L2 = {l2:.3f}")
            axes[0, column].set_ylabel("Waveform (1e-6)")
            axes[0, column].legend(loc="upper right", fontsize=8)
            axes[1, column].plot(t, prediction - fd, color="tab:red", linewidth=1)
            axes[1, column].axhline(0, color="black", linewidth=0.6)
            axes[1, column].set_xlabel("Time (s)")
            axes[1, column].set_ylabel("Model - FD (1e-6)")
            for axis in axes[:, column]:
                axis.grid(alpha=0.2)
                axis.set_xlim(0, 2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
