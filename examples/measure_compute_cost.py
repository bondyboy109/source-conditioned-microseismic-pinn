"""Summarize measured PINN, FMM and FD costs for the paper.

The script only reads completed run summaries and comparison results.  It does
not retrain a model, regenerate observations, or tune an inversion rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a non-empty finite measurement vector")
    return float(values.mean())


def _ceil_positive(value: float) -> int | None:
    if not math.isfinite(value) or value <= 0:
        return None
    return int(math.ceil(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(summary_paths: list[Path], comparison_path: Path) -> dict:
    if not summary_paths:
        raise ValueError("At least one training summary is required")
    summaries = [_read(path) for path in summary_paths]
    comparison = _read(comparison_path)
    run = summaries[-1]["run"]
    stages = [summary["5Hz"] for summary in summaries]
    events = comparison["events"]
    if not events:
        raise ValueError("Comparison file contains no events")

    pinn_seconds = [row["pinn"]["runtime_seconds"] for row in events]
    fmm_seconds = [row["fmm"]["search_seconds"] for row in events
                   if row["fmm"].get("status") == "ok"]
    fd_seconds = [row["fd_runtime_s"] for row in events]
    if not fmm_seconds:
        raise ValueError("Comparison file contains no successful FMM searches")

    training_records = len(run.get("receiver_observations", []))
    training_seconds = float(sum(stage["seconds"] for stage in stages))
    fd_mean = _mean(fd_seconds)
    pinn_mean = _mean(pinn_seconds)
    fmm_mean = _mean(fmm_seconds)
    fmm_setup = float(comparison["fmm_setup_seconds"])
    pinn_load = float(comparison.get("pinn_setup_seconds", 0.0))

    # These are arithmetic crossovers, not accuracy claims.  They answer the
    # narrow question: when does measured one-time cost get amortized?
    pinn_vs_fd = training_seconds / (fd_mean - pinn_mean)
    fmm_vs_fd = fmm_setup / (fd_mean - fmm_mean)
    pinn_fixed = training_seconds + pinn_load
    if fmm_mean < pinn_mean and fmm_setup <= pinn_fixed:
        fmm_cheaper_event = 1
    elif fmm_mean < pinn_mean:
        fmm_cheaper_event = _ceil_positive(
            (fmm_setup - pinn_fixed) / (pinn_mean - fmm_mean))
    else:
        fmm_cheaper_event = None
    # If the training records themselves are counted as data-generation work,
    # include one FD solve per supervised source in the fixed PINN cost.
    end_to_end_fixed = pinn_fixed + training_records * fd_mean
    end_to_end_pinn_vs_fd = end_to_end_fixed / (fd_mean - pinn_mean)

    expected_checkpoint_hash = comparison.get("pinn_checkpoint_sha256")
    if expected_checkpoint_hash is not None:
        final_checkpoint = Path(stages[-1]["checkpoint"])
        if not final_checkpoint.exists():
            raise FileNotFoundError(final_checkpoint)
        if _sha256(final_checkpoint) != expected_checkpoint_hash:
            raise ValueError("Final training checkpoint does not match comparison")

    return {
        "inputs": {
            "training_summaries": [str(path) for path in summary_paths],
            "comparison": str(comparison_path),
            "comparison_event_count": len(events),
            "training_record_count": training_records,
            "device": run.get("device"),
            "torch_version": run.get("torch_version"),
            "numpy_version": run.get("numpy_version"),
            "seed": run.get("seed"),
            "parameter_count": run.get("parameter_count"),
            "checkpoint_sha256": expected_checkpoint_hash,
        },
        "measured": {
            "pinn_training_seconds": training_seconds,
            "pinn_training_stage_seconds": [float(stage["seconds"])
                                             for stage in stages],
            "pinn_checkpoint_load_seconds": pinn_load,
            "pinn_inverse_seconds_per_event_mean": pinn_mean,
            "pinn_inverse_seconds_per_event_median": float(np.median(pinn_seconds)),
            "pinn_peak_cuda_mib": float(max(
                stage["peak_cuda_memory_bytes"] for stage in stages) / 2**20),
            "fmm_setup_seconds": fmm_setup,
            "fmm_search_seconds_per_event_mean": fmm_mean,
            "fmm_search_seconds_per_event_median": float(np.median(fmm_seconds)),
            "fmm_cache_mib": float(comparison["fmm_receiver_fields_mib"]),
            "fd_forward_seconds_per_event_mean": fd_mean,
            "fd_forward_seconds_per_event_median": float(np.median(fd_seconds)),
        },
        "break_even_events": {
            "pinn_vs_fd_inference_only": _ceil_positive(pinn_vs_fd),
            "fmm_vs_fd_inference_only": _ceil_positive(fmm_vs_fd),
            "pinn_per_event_faster_than_fmm": pinn_mean < fmm_mean,
            "fmm_becomes_cheaper_than_pinn_at_event": fmm_cheaper_event,
            "pinn_vs_fd_including_training_record_generation":
                _ceil_positive(end_to_end_pinn_vs_fd),
        },
        "raw_event_seconds": {
            "pinn_inverse": [float(value) for value in pinn_seconds],
            "fmm_search": [float(value) for value in fmm_seconds],
            "fd_forward": [float(value) for value in fd_seconds],
        },
        "interpretation": {
            "scope": "Arithmetic cost amortization on the recorded RTX 4060 Laptop GPU run; no accuracy equivalence is implied.",
            "fd_generation": "FD times are forward-data generation measurements. They are not a matched FD inverse implementation.",
            "pinn_vs_fmm": "PINN has the smaller fixed setup cost, but FMM has the smaller per-event cost. The reported crossover is the first event count at which FMM becomes cheaper in total.",
        },
    }


def render_markdown(result: dict) -> str:
    m = result["measured"]
    b = result["break_even_events"]
    i = result["inputs"]
    lines = [
        "# Measured computational cost and break-even analysis",
        "",
        "This report aggregates completed measurements; it does not retrain or retune a model.",
        "",
        "| Quantity | Measured value |",
        "|---|---:|",
        f"| Device | {i['device']} (PyTorch {i['torch_version']}) |",
        f"| Trainable parameters | {i['parameter_count']:,} |",
        f"| PINN training time | {m['pinn_training_seconds']:.2f} s |",
        f"| PINN peak CUDA allocation | {m['pinn_peak_cuda_mib']:.1f} MiB |",
        f"| PINN inverse time per event | {m['pinn_inverse_seconds_per_event_mean']:.3f} s mean; {m['pinn_inverse_seconds_per_event_median']:.3f} s median |",
        f"| FMM field setup | {m['fmm_setup_seconds']:.2f} s |",
        f"| FMM search time per event | {m['fmm_search_seconds_per_event_mean']:.3f} s mean; {m['fmm_search_seconds_per_event_median']:.3f} s median |",
        f"| FMM cached receiver fields | {m['fmm_cache_mib']:.1f} MiB |",
        f"| FD forward time per event | {m['fd_forward_seconds_per_event_mean']:.3f} s mean; {m['fd_forward_seconds_per_event_median']:.3f} s median |",
        "",
        "## Arithmetic crossover",
        "",
        "The crossover is the smallest event count at which the accumulated measured cost is no larger than the comparator. It is a cost calculation, not an accuracy comparison.",
        "",
        "| Comparison | Events to amortize |",
        "|---|---:|",
        f"| PINN vs FD, inference only | {b['pinn_vs_fd_inference_only'] or 'no positive crossover'} |",
        f"| FMM vs FD, inference only | {b['fmm_vs_fd_inference_only'] or 'no positive crossover'} |",
        f"| FMM becomes cheaper in total than PINN | {b['fmm_becomes_cheaper_than_pinn_at_event'] or 'no positive crossover'} |",
        f"| PINN vs FD, including FD generation for {i['training_record_count']} supervised records | {b['pinn_vs_fd_including_training_record_generation'] or 'no positive crossover'} |",
        "",
        "The PINN per-event inverse is slower than the cached-field FMM search in this run. FD timings here measure forward-record generation, not a complete matched FD inversion. The accuracy results must therefore be reported separately from this arithmetic cost analysis.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-summaries", type=Path, nargs="+",
        default=[
            ROOT / "outputs/depth_gap_25_receiver_5000/training_summary.json",
            ROOT / "outputs/depth_gap_25_hybrid_1000/training_summary.json",
        ],
    )
    parser.add_argument(
        "--comparison", type=Path,
        default=ROOT / "experiments/cross_region_fmm_comparison.json",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "experiments/computational_cost_results.json",
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=ROOT / "experiments/computational_cost_results.md",
    )
    args = parser.parse_args()
    result = collect(args.training_summaries, args.comparison)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
