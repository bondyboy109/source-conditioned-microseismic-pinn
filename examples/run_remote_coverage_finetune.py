"""Fine-tune the selected checkpoint with four previously inspected remote records."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.audit_eikonal_training_generalization import evaluate
from examples.run_fmm_seeded_waveform_development import hash_file
from receiver_training import ReceiverTrainingData
from source_conditioned_pinn import SourceConditionedConfig, load_conditioned_checkpoint, save_conditioned_checkpoint

PROTOCOL = ROOT / "experiments/remote_coverage_finetune_protocol.json"


def main():
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    base_path = ROOT / spec["base_checkpoint"]
    output_dir = ROOT / spec["output_directory"]
    result_path = ROOT / spec["result_file"]
    if output_dir.exists() or result_path.exists():
        raise FileExistsError("Remote fine-tuning output exists; refusing to overwrite")
    if hash_file(base_path) != spec["base_checkpoint_sha256"]:
        raise ValueError("Selected base checkpoint changed")
    remote_paths = [ROOT / row["path"] for row in spec["remote_records"]]
    remote_hashes = []
    for path, row in zip(remote_paths, spec["remote_records"]):
        digest = hash_file(path)
        if row["sha256"] and digest != row["sha256"]:
            raise ValueError(f"Remote record changed: {path}")
        remote_hashes.append(digest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, metadata = load_conditioned_checkpoint(base_path, device)
    if metadata["frequency_hz"] != 5:
        raise ValueError("Expected a 5 Hz base checkpoint")
    original_records = metadata["training_record"]["receiver_observations"]
    original_paths = [Path(item["path"]) for item in original_records]
    if len(original_records) != 33:
        raise ValueError("Expected the 33-source base manifest")
    for item, path in zip(original_records, original_paths):
        if hash_file(path) != item["sha256"]:
            raise ValueError(f"Base training record changed: {path}")
    all_paths = original_paths + remote_paths
    config = SourceConditionedConfig(**metadata["config"])
    train_data = ReceiverTrainingData(all_paths, device, config, 5., seed=spec["seed"])
    control = copy.deepcopy(base).to(device)
    candidate = copy.deepcopy(base).to(device)
    del base
    optimizer = torch.optim.Adam(candidate.parameters(), lr=spec["learning_rate"])
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec["seed"])
    history = []
    started = time.perf_counter()
    for update in range(1, spec["updates"] + 1):
        points, target, source, rms = train_data.batch(
            spec["receiver_batch_size"], spec["receiver_events_per_batch"])
        candidate.train()
        optimizer.zero_grad()
        receiver = ((candidate(points, source).squeeze(-1) / rms - target) ** 2).mean()
        total = spec["receiver_loss_weight"] * receiver
        total.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
        optimizer.step()
        history.append({"receiver": float(receiver.detach()), "total": float(total.detach())})
        if update == 1 or update % 100 == 0:
            print(f"update {update}/{spec['updates']}: receiver={history[-1]['receiver']:.5f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    scale = np.asarray([metadata["config"][key] for key in
                        ("x_length_m", "y_length_m", "z_length_m", "time_length_s")], dtype=np.float32)
    remote_rows = []
    for path in remote_paths:
        remote_rows.append({"event": path.name, "sha256": hash_file(path),
                            "control": evaluate(control, scale, path, hash_file(path)),
                            "candidate": evaluate(candidate, scale, path, hash_file(path))})
    local_dir = ROOT / "data/observations/midpoint_teacher_confirmation"
    local_rows = []
    for path in sorted(local_dir.glob("local_*.npz")):
        digest = hash_file(path)
        local_rows.append({"event": path.name, "sha256": digest,
                           "control": evaluate(control, scale, path, digest),
                           "candidate": evaluate(candidate, scale, path, digest)})
    depth = json.loads((ROOT / "experiments/eikonal_depth_sweep_results.json").read_text())
    depth_rows = []
    for item in depth["events"]:
        path = ROOT / "data/observations/eikonal_depth_sweep" / item["file"]
        digest = hash_file(path)
        depth_rows.append({"event": path.name, "sha256": digest,
                           "control": evaluate(control, scale, path, digest),
                           "candidate": evaluate(candidate, scale, path, digest)})
    training_record = copy.deepcopy(metadata["training_record"])
    training_record["receiver_observations"] = train_data.metadata
    training_record["remote_coverage_finetune"] = {
        "role": spec["role"], "remote_records": remote_hashes,
        "updates": spec["updates"], "learning_rate": spec["learning_rate"],
        "independent_validation": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    save_conditioned_checkpoint(output_dir / "candidate_loadable.pth", candidate, None,
                                metadata["frequency_hz"], spec["updates"], config,
                                history=history[-1:], training_record=training_record)
    summary = {
        "remote_control_mean_full_l2": float(np.mean([r["control"]["full_record_relative_l2"] for r in remote_rows])),
        "remote_candidate_mean_full_l2": float(np.mean([r["candidate"]["full_record_relative_l2"] for r in remote_rows])),
        "local_control_mean_full_l2": float(np.mean([r["control"]["full_record_relative_l2"] for r in local_rows])),
        "local_candidate_mean_full_l2": float(np.mean([r["candidate"]["full_record_relative_l2"] for r in local_rows])),
        "depth_control_mean_full_l2": float(np.mean([r["control"]["full_record_relative_l2"] for r in depth_rows])),
        "depth_candidate_mean_full_l2": float(np.mean([r["candidate"]["full_record_relative_l2"] for r in depth_rows])),
    }
    result = {"role": spec["role"], "base_checkpoint_sha256": spec["base_checkpoint_sha256"],
              "remote_record_sha256": remote_hashes, "elapsed_seconds": elapsed,
              "history_tail": history[-1], "summary": summary,
              "remote_events": remote_rows, "local_events": local_rows,
              "depth_events": depth_rows,
              "candidate_checkpoint": str((output_dir / "candidate_loadable.pth").resolve()),
              "limits": spec["limits"]}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {result_path}", flush=True)
    print(f"summary={summary}", flush=True)


if __name__ == "__main__":
    main()
