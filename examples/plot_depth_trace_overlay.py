"""Render independent FD and PINN traces and their 80 m depth responses."""

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
    with np.load(args.traces) as archive:
        times = archive["times_s"]
        indices = archive["receiver_indices"]
        receivers = archive["receivers_m"]
        fd860, pinn860 = archive["fd_860"] * 1e6, archive["pinn_860"] * 1e6
        fd940, pinn940 = archive["fd_940"] * 1e6, archive["pinn_940"] * 1e6
    fig, axes = plt.subplots(len(indices), 3, figsize=(14, 10), sharex=True)
    for row, (index, xyz) in enumerate(zip(indices, receivers)):
        series = ((fd860[row], pinn860[row]), (fd940[row], pinn940[row]),
                  (fd940[row] - fd860[row], pinn940[row] - pinn860[row]))
        for column, (fd, pinn) in enumerate(series):
            axis = axes[row, column]
            axis.plot(times, fd, color="black", linewidth=1.2, label="FD")
            axis.plot(times, pinn, color="tab:orange", linewidth=1.0, label="PINN")
            axis.axhline(0, color="0.65", linewidth=.6)
            axis.set_xlim(.4, 2.)
            axis.grid(alpha=.2)
            if column == 0:
                axis.set_ylabel(f"R{index}: ({int(xyz[0])}, {int(xyz[1])}) m\nAmplitude (1e-6)")
            if row == len(indices) - 1:
                axis.set_xlabel("Time (s)")
            if row == 0:
                axis.set_title(("Training: z=860 m", "Development: z=940 m",
                                "Response: z=940 minus 860 m")[column])
            if row == 0 and column == 2:
                axis.legend(loc="upper right", frameon=False)
    fig.suptitle("Frozen 5 Hz PINN vs FD at the four largest-error stations", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=175)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
