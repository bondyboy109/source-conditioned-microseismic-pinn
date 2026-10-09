"""Compare remote-coverage fine-tuning on 16 spent nontraining FD records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_training_generalization import evaluate
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/remote_coverage_generalization_audit_protocol.json"


def historical_events(spec: dict):
    rows = []
    for group_index, item in enumerate(spec["historical_results"], start=1):
        result_path = ROOT / item["path"]
        if hash_file(result_path) != item["sha256"]:
            raise ValueError(f"Historical result changed: {result_path}")
        old = json.loads(result_path.read_text(encoding="utf-8"))
        for event in old["events"]:
            path = (Path(event["observation"]) if "observation" in event
                    else ROOT / item["data_directory"] / event["event"])
            digest = event.get("observation_sha256", event.get("sha256"))
            if hash_file(path) != digest:
                raise ValueError(f"Historical observation changed: {path}")
            rows.append((f"group_{group_index}", path, digest))
    if len(rows) != 16 or len({path.resolve() for _, path, _ in rows}) != 16:
        raise ValueError("Expected 16 distinct historical observations")
    return rows


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    output = ROOT / spec["result_file"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for label in ("base", "candidate"):
        if hash_file(ROOT / spec[f"{label}_checkpoint"]) != spec[f"{label}_sha256"]:
            raise ValueError(f"{label} checkpoint changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for label in ("base", "candidate"):
        model, metadata = load_conditioned_checkpoint(ROOT / spec[f"{label}_checkpoint"], device)
        model.eval().requires_grad_(False)
        scale = np.asarray([metadata["config"][key] for key in
                            ("x_length_m", "y_length_m", "z_length_m", "time_length_s")],
                           dtype=np.float32)
        loaded[label] = (model, scale, metadata)
    added = loaded["candidate"][2]["training_record"]["receiver_observations"][33:]
    if len(added) != 4:
        raise ValueError("Candidate does not contain four added remote records")
    added_hashes = {row["sha256"] for row in added}
    added_sources = np.asarray([row["source_truth"][:3] for row in added], dtype=np.float64)
    rows = []
    for group, path, digest in historical_events(spec):
        if digest in added_hashes:
            raise ValueError(f"Training/evaluation content overlap: {path}")
        with np.load(path) as archive:
            source = np.asarray(archive["source_truth"], dtype=np.float64)
        row = {"group": group, "event": path.name, "sha256": digest,
               "source_m_s": source.tolist(),
               "nearest_added_remote_distance_m": float(np.min(np.linalg.norm(
                   added_sources - source[:3], axis=1)))}
        for label in ("base", "candidate"):
            model, scale, _ = loaded[label]
            row[label] = evaluate(model, scale, path, digest)
        row["candidate_improves"] = (row["candidate"]["full_record_relative_l2"]
                                     < row["base"]["full_record_relative_l2"])
        rows.append(row)
        print(f"{path.name}: base={row['base']['full_record_relative_l2']:.3f}, "
              f"candidate={row['candidate']['full_record_relative_l2']:.3f}", flush=True)
    summary = {}
    for label in ("base", "candidate"):
        summary[f"{label}_mean_full_l2"] = float(np.mean([
            row[label]["full_record_relative_l2"] for row in rows]))
        summary[f"{label}_median_full_l2"] = float(np.median([
            row[label]["full_record_relative_l2"] for row in rows]))
        summary[f"{label}_mean_active_l2"] = float(np.mean([
            row[label]["active_sample_relative_l2"] for row in rows]))
    summary["candidate_improves_count"] = int(sum(row["candidate_improves"] for row in rows))
    result = {"role": spec["role"], "protocol_sha256": hash_file(PROTOCOL),
              "base_sha256": spec["base_sha256"],
              "candidate_sha256": spec["candidate_sha256"],
              "summary": summary, "events": rows, "limits": spec["limits"]}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"summary={summary}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
