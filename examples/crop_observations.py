"""Create registered time-window views of existing FD observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--end-s", type=float, default=2.0)
    args = parser.parse_args()
    if args.end_s <= 0:
        parser.error("--end-s must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in args.input:
        with np.load(source_path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float32)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
            source = np.asarray(archive["source_truth"], dtype=np.float32)
        mask = times <= args.end_s + 1e-6
        if mask.sum() < 2 or not np.isclose(times[mask][-1], args.end_s, atol=1e-5):
            raise ValueError(f"Time axis does not reach exactly {args.end_s:g} s: {source_path}")
        output = args.output_dir / f"{source_path.stem}_0to{args.end_s:g}s.npz"
        np.savez_compressed(output, data=data[:, mask], times=times[mask],
                            receiver_coords=receivers, source_truth=source)
        report = json.loads(source_path.with_suffix(".json").read_text(encoding="utf-8"))
        report.update({"derived_from": str(source_path.resolve()),
                       "time_window_s": [0.0, float(args.end_s)],
                       "duration_s": float(args.end_s),
                       "n_time_samples": int(mask.sum()),
                       "waveform_peak_abs": float(np.max(np.abs(data[:, mask])))})
        output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {output} {data[:, mask].shape}", flush=True)


if __name__ == "__main__":
    main()
