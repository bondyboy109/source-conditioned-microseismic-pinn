"""Plot frozen control and one-source-augmented 5 Hz depth scans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("augmented", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))["events"]
    augmented = json.loads(args.augmented.read_text(encoding="utf-8"))["events"]
    if len(control) != 3 or len(augmented) != 3:
        raise ValueError("Expected three registered development events")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for axis, original, added in zip(axes, control, augmented):
        if Path(original["observation"]).name != Path(added["observation"]).name:
            raise ValueError("Control and augmented source ordering differs")
        z_true = original["scan"]["true_z_m"]
        for row, label, color in ((original, "24 sources", "#444444"),
                                  (added, "25 sources", "#d35c28")):
            scan = row["scan"]
            z = np.asarray(scan["z_m"])
            loss = np.asarray(scan["normalized_mse"])
            axis.plot(z, loss, label=label, color=color, linewidth=1.9)
            axis.scatter(scan["best_z_m"], scan["best_loss"],
                         marker="o", color=color, s=28, zorder=3)
        axis.axvline(z_true, color="#26807a", linewidth=1, linestyle="--",
                     label="FD true depth" if axis is axes[0] else None)
        axis.set_title(f"Development event z={z_true:g} m")
        axis.set_xlabel("Candidate depth (m)")
        axis.set_ylabel("Normalized waveform MSE" if axis is axes[0] else "")
        axis.grid(alpha=.22)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
