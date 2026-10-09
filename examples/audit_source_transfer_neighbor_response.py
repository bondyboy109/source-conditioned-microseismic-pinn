"""Compare frozen transfer models with FD differences between training sources."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_source_transfer_moe import PROTOCOL as PARENT, digest, new_model, read_setup
from examples.run_traveltime_aligned_moe_development import load_observations

PROTOCOL = ROOT / "experiments/source_transfer_neighbor_response_protocol.json"


def response(model, event, receivers, times, domain, stride, device):
    sampled_times = times[::stride]
    nr, nt = len(receivers), len(sampled_times)
    points = np.empty((nr, nt, 4), dtype=np.float32)
    points[:, :, :3] = receivers[:, None, :]
    points[:, :, 3] = sampled_times[None, :]
    points = points.reshape(-1, 4) / domain
    source = torch.as_tensor(event["source"] / domain, device=device).expand(8192, -1)
    indices = np.repeat(np.arange(nr), nt)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(points), 8192):
            end = min(start + 8192, len(points))
            x = torch.as_tensor(points[start:end], device=device)
            receiver_id = torch.as_tensor(indices[start:end], device=device)
            predictions.append(model(x, source[:end - start], receiver_id)[0]
                               .squeeze(1).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent, setup, _, paths = read_setup()
    if ROOT / spec["parent_protocol"] != PARENT:
        raise ValueError("Parent protocol changed")
    if spec["arms"] != parent["arms"][:2] or spec["seeds"] != parent["seeds"]:
        raise ValueError("Arms or seeds changed")
    result_file = ROOT / spec["result_json"]
    report_file = ROOT / spec["result_report"]
    if result_file.exists() or report_file.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    events, receivers, times = load_observations(paths)
    for event in events:
        report = json.loads(event["path"].with_suffix(".json").read_text(encoding="utf-8"))
        if report["method"] != "second_order_3d_acoustic_FD" or report["frequency_hz"] != 5.0:
            raise ValueError(f"Incompatible FD acquisition: {event['path']}")
    pairs = []
    for a in range(len(events)):
        for b in range(a + 1, len(events)):
            distance = float(np.linalg.norm(events[a]["source"][:3]
                                            - events[b]["source"][:3]))
            dt = float(abs(events[a]["source"][3] - events[b]["source"][3]))
            if spec["minimum_distance_m"] <= distance <= spec["maximum_distance_m"] \
                    and dt <= spec["maximum_origin_time_difference_s"]:
                pairs.append({"a": a, "b": b, "separation_m": distance})
    if not pairs:
        raise ValueError("No registered matched-time pairs")
    targets = [(e["data"][:, ::spec["time_stride"]] / e["rms"])
               .astype(np.float64).reshape(-1) for e in events]
    for pair in pairs:
        delta = targets[pair["b"]] - targets[pair["a"]]
        pair["fd_delta_rms"] = float(np.sqrt(np.mean(delta ** 2)))
        pair["fd_delta_l2"] = float(np.linalg.norm(delta))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fields = np.load(ROOT / parent["target_fmm_fields"], mmap_mode="r")
    domain = np.asarray([*setup["model"]["spatial_lengths_m"],
                         setup["model"]["time_length_s"]], dtype=np.float32)
    rows = []
    for arm in spec["arms"]:
        for seed in spec["seeds"]:
            checkpoint = ROOT / "outputs/source_transfer_moe" / f"{arm}_seed{seed}.pth"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (payload["protocol_sha256"] != digest(PARENT)
                    or payload["target_fields_sha256"]
                    != digest(ROOT / parent["target_fmm_fields"])
                    or payload["arm"] != arm or payload["seed"] != seed):
                raise ValueError(f"Checkpoint provenance mismatch: {checkpoint}")
            model = new_model(fields, receivers, setup, device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            predicted = [response(model, e, receivers, times, domain,
                                  spec["time_stride"], device) for e in events]
            for pair in pairs:
                a, b = pair["a"], pair["b"]
                actual = targets[b] - targets[a]
                estimate = predicted[b] - predicted[a]
                actual_norm = pair["fd_delta_l2"]
                predicted_norm = float(np.linalg.norm(estimate))
                rows.append({"arm": arm, "seed": seed, "a": a, "b": b,
                             "amplitude_ratio": predicted_norm / actual_norm,
                             "delta_relative_l2": float(np.linalg.norm(estimate - actual)
                                                        / actual_norm),
                             "delta_cosine": float(np.dot(estimate, actual)
                                                   / max(predicted_norm * actual_norm, 1e-30)),
                             "fd_delta_rms": pair["fd_delta_rms"]})
            print(f"{arm} seed={seed}: {len(pairs)} training-source pairs", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {}
    for arm in spec["arms"]:
        subset = [r for r in rows if r["arm"] == arm]
        summary[arm] = {metric: float(np.median([r[metric] for r in subset]))
                        for metric in ("amplitude_ratio", "delta_relative_l2", "delta_cosine")}
        summary[arm]["pair_seed_comparisons"] = len(subset)
    result = {"protocol_sha256": digest(PROTOCOL), "parent_protocol_sha256": digest(PARENT),
              "device": str(device), "pair_count": len(pairs), "event_count": len(events),
              "samples_per_trace": len(times[::spec["time_stride"]]),
              "pairs": pairs, "rows": rows, "summary": summary,
              "restriction": spec["restriction"]}
    lines = ["# Neighboring source waveform response audit", "",
             f"{len(pairs)} matched-time FD training-source pairs among {len(events)} events; "
             f"{len(times[::spec['time_stride']])} samples per receiver trace. "
             "Both predictions and FD records are evaluated at identical receiver/time coordinates.", "",
             "| Training path | Median predicted/FD difference amplitude | "
             "Median relative L2 of difference | Median difference cosine |",
             "|---|---:|---:|---:|"]
    for arm in spec["arms"]:
        row = summary[arm]
        lines.append(f"| {arm} | {row['amplitude_ratio']:.3f} | "
                     f"{row['delta_relative_l2']:.3f} | {row['delta_cosine']:.3f} |")
    lines.extend(["", "Each median is over matched pairs and three frozen seeds. "
                  "A relative L2 below 1 improves on a zero predicted difference; "
                  "a cosine near zero means the predicted change is poorly aligned with FD. "
                  "Per-pair and per-seed measurements are in the JSON file.", "",
                  spec["restriction"]])
    result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
