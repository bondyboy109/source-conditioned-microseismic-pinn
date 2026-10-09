"""Create a weight-identical checkpoint with the registered onset gate enabled."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.run_fmm_seeded_waveform_development import hash_file
from source_conditioned_pinn import load_conditioned_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    protocol = json.loads(args.protocol.read_text())
    gate = protocol["onset_gate"]
    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    if payload["architecture"]["kind"] != "eikonal_travel_time_carrier":
        raise ValueError("Only the frozen eikonal carrier is supported")
    if payload["architecture"].get("onset_gate") is not None:
        raise ValueError("Source checkpoint already has an onset gate")
    original_state = {key: value.clone() for key, value in payload["model_state_dict"].items()}
    payload["architecture"]["onset_gate"] = gate
    payload["derived_operator"] = {
        "operation": "enable fixed differentiable onset gate; weights unchanged",
        "source_checkpoint": str(args.source.resolve()),
        "source_sha256": hash_file(args.source),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": hash_file(args.protocol),
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.destination)
    restored, restored_payload = load_conditioned_checkpoint(args.destination,
                                                              torch.device("cpu"))
    for key, value in original_state.items():
        if not torch.equal(value, restored_payload["model_state_dict"][key]):
            raise ValueError(f"Model weight changed: {key}")
    if restored.onset_gate != gate:
        raise ValueError("Derived checkpoint did not restore the registered gate")
    print(json.dumps({"source_sha256": hash_file(args.source),
                      "derived_sha256": hash_file(args.destination),
                      "onset_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
