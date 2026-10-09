"""Export the selected teacher-weight state as a standard loadable checkpoint."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import (
    SourceConditionedConfig, load_conditioned_checkpoint, save_conditioned_checkpoint,
)

BASE = ROOT / "outputs/eikonal_depth_coverage33_hybrid_1000/conditioned_5Hz.pth"
STATE = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002.pth"
PROTOCOL = ROOT / "experiments/depth_midpoint_teacher_low_weight_protocol.json"
RESULT = ROOT / "experiments/depth_midpoint_teacher_low_weight_results.json"
OUTPUT = ROOT / "outputs/depth_midpoint_teacher_low_weight/teacher_0.002_loadable.pth"
BASE_SHA256 = "3aa5ade41968a0d9e1a214c625ac4a62c075eac23bea17b7c60cc8b7d3751ab9"
STATE_SHA256 = "0c9406233ee7ff1ee116507483c10375f72a5b98b5b62cbbabe03dc5e05b4064"


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if hash_file(BASE) != BASE_SHA256 or hash_file(STATE) != STATE_SHA256:
        raise ValueError("Selected model input changed")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if (result["selected_label"] != "teacher_0.002"
            or result["protocol_sha256"] != hash_file(PROTOCOL)):
        raise ValueError("Selected result provenance changed")
    device = torch.device("cpu")
    model, metadata = load_conditioned_checkpoint(BASE, device)
    state = torch.load(STATE, map_location=device, weights_only=False)
    if state.get("label") != "teacher_0.002" or state.get("teacher_loss_weight") != 0.002:
        raise ValueError("Unexpected selected state")
    model.load_state_dict(state["model_state_dict"])
    training_record = copy.deepcopy(metadata["training_record"])
    training_record["development_midpoint_teacher"] = {
        "role": result["role"],
        "teacher_loss_weight": 0.002,
        "updates": state["updates"],
        "protocol_sha256": result["protocol_sha256"],
        "independent_validation": False,
    }
    save_conditioned_checkpoint(
        OUTPUT, model, None, metadata["frequency_hz"], state["updates"],
        SourceConditionedConfig(**metadata["config"]),
        history=[result["history_tail"]["teacher_0.002"]],
        training_record=training_record,
    )
    _, exported = load_conditioned_checkpoint(OUTPUT, device)
    if exported["training_record"]["development_midpoint_teacher"]["teacher_loss_weight"] != 0.002:
        raise ValueError("Exported checkpoint metadata verification failed")
    print(f"Saved and reloaded {OUTPUT}")
    print(f"sha256={hash_file(OUTPUT)}")


if __name__ == "__main__":
    main()
