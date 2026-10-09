"""Post-selection waveform sanity check at known sources versus zero output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.validate_arrival_constrained import digest, load_model, load_protocol, waveform_losses

COMPARISON = ROOT / "experiments/refined_fmm_moe_diagnostic_results.json"
EXPECTED_COMPARISON_SHA256 = "91ced7c91acb7de938ddff9009fbd1f0380ab8fe1c2cbebc7baa457722b55071"
OUTPUT = ROOT / "experiments/refined_moe_truth_waveform_audit.json"


def zero_waveform_loss(data):
    """Compute the zero-predictor MSE with exactly the waveform scorer's normalization."""
    values = data.astype(np.float64)
    return float(np.mean(values[::2, ::8] ** 2) / np.mean(values ** 2))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if digest(COMPARISON) != EXPECTED_COMPARISON_SHA256:
        raise ValueError("Frozen MoE comparison changed")
    result = json.loads(COMPARISON.read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "experiments/arrival_spatial_refinement_protocol.json")
                          .read_text(encoding="utf-8"))
    spec = load_protocol()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(spec, device)
    rows = []
    for scored in result["events"]:
        path = ROOT / protocol["test_directory"] / (scored["id"] + ".npz")
        if digest(path) != scored["observation_sha256"]:
            raise ValueError(f"FD observation changed: {path}")
        with np.load(path) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        baseline = zero_waveform_loss(data)
        at_truth = float(waveform_losses(model, truth[None, :], data, times,
                                         receivers, device)[0])
        near = scored["methods"]["refined_fmm"]["waveform_loss"]
        rows.append({"id": scored["id"], "zero_prediction_loss": baseline,
                     "moe_at_true_source_loss": at_truth, "moe_at_fmm_source_loss": near,
                     "true_source_better_than_zero": bool(at_truth < baseline),
                     "fmm_source_better_than_zero": bool(near < baseline)})
        print(f"{scored['id']}: zero={baseline:.3f}, MoE(true)={at_truth:.3f}, "
              f"MoE(FMM)={near:.3f}", flush=True)
    payload = {"role": "post-selection diagnostic using known truth; never used for inversion",
               "comparison_sha256": EXPECTED_COMPARISON_SHA256,
               "summary": {"mean_zero_loss": float(np.mean([r["zero_prediction_loss"] for r in rows])),
                           "mean_moe_true_loss": float(np.mean([r["moe_at_true_source_loss"] for r in rows])),
                           "moe_true_better_than_zero_count": sum(r["true_source_better_than_zero"] for r in rows),
                           "moe_fmm_better_than_zero_count": sum(r["fmm_source_better_than_zero"] for r in rows)},
               "events": rows}
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
