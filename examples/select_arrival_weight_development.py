"""Select the waveform/arrival objective weight on the registered development events."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import skfmm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))
from audit_fmm_traveltime import first_arrival, trilinear
from mixture_of_experts import PrototypeGatedResidualMixtureOfExperts
from source_conditioned_pinn import load_conditioned_checkpoint

BASE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
MOE = ROOT / "outputs/prototype_moe_development/joint_moe_physics.pth"
RESULTS = (
    (ROOT / "experiments/eikonal_independent_results.json", ROOT / "data/observations/eikonal_independent"),
    (ROOT / "experiments/eikonal_wide_search_independent_results.json", ROOT / "data/observations/eikonal_wide_search_independent"),
)
OUT = ROOT / "experiments/prototype_moe_arrival_weight_development.json"
DOMAIN = np.asarray([16000., 16000., 3720., 9.], dtype=np.float32)
BOUNDS = np.asarray([[2000., 14000.], [2000., 14000.], [100., 1500.], [0.05, 0.5]], dtype=np.float32)
SPACING = 160.0
LOCAL_STEP = np.asarray([2400., 2400., 466.6666667], dtype=np.float32)
ALPHAS = (0.10, 0.25, 0.50, 1.00)


def fmm_arrivals(velocity, xyz, receivers):
    index = np.clip(np.rint(xyz / SPACING).astype(int), 0, np.asarray(velocity.shape) - 1)
    level = np.ones(velocity.shape, dtype=np.float64)
    level[tuple(index)] = -1.0
    field = np.asarray(skfmm.travel_time(level, speed=velocity, dx=SPACING), dtype=np.float64)
    return trilinear(field, receivers, SPACING)


def waveform_prediction(model, source_m_s, receiver_norm, time_norm, device):
    points = np.concatenate((
        np.broadcast_to(receiver_norm[:, None, :], (len(receiver_norm), len(time_norm), 3)),
        np.broadcast_to(time_norm[None, :, None], (len(receiver_norm), len(time_norm), 1)),
    ), axis=-1).reshape(-1, 4)
    source = torch.as_tensor(source_m_s / DOMAIN, dtype=torch.float32, device=device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(points), 8192):
            x = torch.as_tensor(points[start:start + 8192], dtype=torch.float32, device=device)
            values.append(model(x, source.expand(x.shape[0], -1), freeze_base=True)[0])
    return torch.cat(values).reshape(len(receiver_norm), len(time_norm))


def event_path(row, directory):
    candidate = Path(row.get("observation", ""))
    if not candidate.is_file():
        candidate = directory / row["event"]
    return candidate


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _ = load_conditioned_checkpoint(BASE, device)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    spec = torch.load(MOE, map_location=device, weights_only=False)
    model = PrototypeGatedResidualMixtureOfExperts(
        base, expert_count=int(spec["spec"]["expert_count"]),
        hidden_dim=int(spec["spec"]["expert_hidden_dim"]),
        hidden_layers=int(spec["spec"]["expert_hidden_layers"]),
        feature_scale=float(spec["spec"]["feature_scale"]),
    ).to(device)
    model.load_state_dict(spec["state_dict"])
    model.eval()
    velocity = np.asarray(np.load(ROOT / "data/velocity/vp_model.npy"), dtype=np.float64)[::4, ::4, ::4]
    events = []
    for result_path, directory in RESULTS:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for row in payload["events"]:
            path = event_path(row, directory)
            with np.load(path) as archive:
                data = np.asarray(archive["data"], dtype=np.float32)
                times = np.asarray(archive["times"], dtype=np.float64)
                receivers = np.asarray(archive["receiver_coords"], dtype=np.float64)
                truth = np.asarray(archive["source_truth"], dtype=np.float64)
            fmm_row = row.get("fmm", row)
            center_values = fmm_row.get("estimated_source_m_s") if isinstance(fmm_row, dict) else None
            if center_values is None and isinstance(row.get("fmm"), dict):
                center_values = row["fmm"].get("estimated_source_m_s")
            if center_values is None or len(center_values) < 3:
                print(f"skip {path.name}: no valid FMM starting source", flush=True)
                continue
            center = np.asarray(center_values, dtype=np.float32)[:3]
            picks = np.asarray([first_arrival(trace, times) for trace in data], dtype=np.float64)
            valid = np.isfinite(picks)
            receiver_norm = (receivers / DOMAIN[:3]).astype(np.float32)[::2]
            time_norm = (times / DOMAIN[3]).astype(np.float32)[::8]
            observed = torch.as_tensor(data[::2, ::8], dtype=torch.float32, device=device)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        xyz = np.clip(center + np.asarray([dx, dy, dz], dtype=np.float32) * LOCAL_STEP,
                                       BOUNDS[:3, 0], BOUNDS[:3, 1])
                        travel = fmm_arrivals(velocity, xyz, receivers)
                        origin = float(np.clip(np.median(picks[valid] - travel[valid]), *BOUNDS[3]))
                        arrival_rmse = float(np.sqrt(np.mean((origin + travel[valid] - picks[valid]) ** 2)))
                        source = np.r_[xyz, origin].astype(np.float32)
                        prediction = waveform_prediction(model, source, receiver_norm, time_norm, device)
                        waveform_loss = float(((prediction - observed) / rms).square().mean())
                        arrival_loss = (arrival_rmse / 0.10) ** 2
                        candidates.append({"source_m_s": source.tolist(),
                                           "waveform_loss": waveform_loss,
                                           "arrival_rmse_s": arrival_rmse,
                                           "arrival_normalized_loss": arrival_loss,
                                           **{f"hybrid_alpha_{alpha:g}": waveform_loss + alpha * arrival_loss
                                              for alpha in ALPHAS}})
            event = {"event": path.name, "observation": str(path), "truth_source_m_s": truth.tolist(),
                     "fmm_start_m_s": center.tolist(), "valid_picks": int(valid.sum())}
            for name, key in (("waveform", "waveform_loss"), ("arrival", "arrival_rmse_s")):
                choice = min(candidates, key=lambda value, key=key: value[key])
                estimate = np.asarray(choice["source_m_s"], dtype=np.float64)
                event[f"{name}_location_error_m"] = float(np.linalg.norm(estimate[:3] - truth[:3]))
                event[f"{name}_origin_time_error_s"] = float(abs(estimate[3] - truth[3]))
            for alpha in ALPHAS:
                key = f"hybrid_alpha_{alpha:g}"
                choice = min(candidates, key=lambda value, key=key: value[key])
                estimate = np.asarray(choice["source_m_s"], dtype=np.float64)
                event[f"hybrid_{alpha:g}_location_error_m"] = float(np.linalg.norm(estimate[:3] - truth[:3]))
                event[f"hybrid_{alpha:g}_origin_time_error_s"] = float(abs(estimate[3] - truth[3]))
            events.append(event)
            print(path.name, flush=True)
    summary = {}
    for name in ("waveform", "arrival"):
        summary[name] = {"mean_location_error_m": float(np.mean([row[f"{name}_location_error_m"] for row in events])),
                         "mean_origin_time_error_s": float(np.mean([row[f"{name}_origin_time_error_s"] for row in events]))}
    for alpha in ALPHAS:
        name = f"hybrid_{alpha:g}"
        summary[name] = {"mean_location_error_m": float(np.mean([row[f"{name}_location_error_m"] for row in events])),
                         "mean_origin_time_error_s": float(np.mean([row[f"{name}_origin_time_error_s"] for row in events]))}
    best_alpha = min(ALPHAS, key=lambda alpha: summary[f"hybrid_{alpha:g}"]["mean_location_error_m"])
    result = {"role": "development-only selection of arrival-loss weight",
              "event_count": len(events), "fmm_spacing_m": SPACING,
              "local_step_m": LOCAL_STEP.tolist(), "candidate_count_per_event": 27,
              "hybrid_arrival_scale_s": 0.10, "alphas": list(ALPHAS),
              "selected_alpha_by_development_location_error": best_alpha,
              "summary": summary, "events": events,
              "limits": "Alpha is selected only on these development events. The independent eight-event holdout must be evaluated once with the selected alpha and no retuning."}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected_alpha": best_alpha, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
