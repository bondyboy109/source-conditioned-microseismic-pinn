"""Replay the two neural localization branches on archived confirmation data.

The outputs are current-version installation checks on already-inspected data,
not new independent confirmation or byte-identical historical reproduction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples import run_traveltime_aligned_moe_localization_current as runner

PROTOCOL = ROOT / "experiments/traveltime_aligned_moe_confirmation_protocol.json"
PROJECTION = ROOT / "results/final_neural_ablation.json"
BRANCHES = ("waveform_only", "hybrid")


def preflight() -> tuple[dict, dict, list[Path]]:
    """Check supplied inputs against original hashes and the neural projection."""
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    localization = json.loads(
        (ROOT / spec["localization_protocol"]).read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("model_checkpoint", "model_checkpoint_sha256"),
        ("receiver_fmm_fields", "receiver_fmm_fields_sha256"),
        ("localization_protocol", "localization_protocol_sha256"),
    ):
        if runner.sha256(ROOT / spec[path_key]) != spec[hash_key]:
            raise ValueError("Input changed: " + spec[path_key])
    mapping = json.loads(
        (ROOT / "experiments/protocol_hash_resolution.json").read_text(
            encoding="utf-8"))
    current_path = (ROOT / mapping["current_runner"]).resolve()
    if (current_path != Path(runner.__file__).resolve()
            or runner.sha256(current_path) != mapping["current_runner_sha256"]):
        raise ValueError("Current runner path or checksum changed")

    projected = json.loads(PROJECTION.read_text(encoding="utf-8"))
    records = projected["confirmation"]["events"]
    hashes = {record["event"]: record["sha256"] for record in records}
    expected_events = [row["file"] for row in spec["events"]]
    if (len(records) != len(hashes)
            or set(hashes) != set(expected_events)
            or len(expected_events) != 8):
        raise ValueError("Neural projection must identify all eight protocol events")

    paths = []
    for row in spec["events"]:
        path = ROOT / spec["confirmation_directory"] / row["file"]
        if runner.sha256(path) != hashes[row["file"]]:
            raise ValueError("Observation changed: " + row["file"])
        with np.load(path) as archive:
            times = np.asarray(archive["times"], dtype=np.float64)
            if (archive["data"].shape != (25, 1001)
                    or archive["receiver_coords"].shape != (25, 3)
                    or times.shape != (1001,)
                    or not np.isclose(times[0], 0.0)
                    or not np.isclose(times[-1], 2.0)
                    or not np.allclose(np.diff(times), 0.002, rtol=1e-4,
                                       atol=1e-7)):
                raise ValueError("Final observation dimensions or times changed")
        paths.append(path)
    return spec, localization, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--events", type=int, choices=(1, 8), default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec, localization, paths = preflight()
    print("Current runner, model, cache, protocol, and eight observations verified.",
          flush=True)
    print("This replays inspected data; it is not new independent confirmation. "
          "The historical add81f runner source is unavailable.", flush=True)
    if args.check_inputs:
        return

    output_root = (ROOT / "outputs/reproduction").resolve()
    output = (args.output or output_root /
              f"current_neural_replay_{args.events}_events.json").resolve()
    try:
        output.relative_to(output_root)
    except ValueError:
        parser.error("--output must be inside outputs/reproduction")
    if output.exists():
        raise FileExistsError("Refusing to overwrite replay output: " + str(output))

    first = runner.load_event(paths[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain = runner.load_model(localization, first["receivers"], device)
    bounds = np.asarray(localization["source_bounds_m_s"], dtype=np.float32)
    search = localization["search"]
    arrival_spec = localization["arrival_constraint"]
    candidates = runner.coarse_grid(bounds, search["coarse_grid_xyzt"])
    rows = []
    for path in paths[:args.events]:
        event = runner.load_event(path, first["receivers"])
        _, picks = runner.pick_arrivals(event, arrival_spec)
        if np.isfinite(picks).sum() < arrival_spec["minimum_valid_picks"]:
            raise ValueError("Too few valid arrivals: " + path.name)
        coarse = runner.sampled_waveforms(
            event, domain, search["coarse_receiver_stride"],
            search["coarse_time_stride"], device)
        fine = runner.sampled_waveforms(
            event, domain, search["refine_receiver_stride"],
            search["refine_time_stride"], device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        waveform = runner.waveform_scores(
            model, candidates, domain, coarse, search["candidate_batch"])
        arrival = runner.arrival_scores(
            model, candidates, picks, arrival_spec["arrival_scale_s"],
            search["candidate_batch"])
        branches = {}
        for name, alpha in (("waveform_only", 0.0),
                            ("hybrid", arrival_spec["hybrid_weight"])):
            total = waveform + alpha * arrival
            indices = runner.choose_diverse_starts(
                candidates, total, search["restarts"],
                search["minimum_start_separation_m"])
            starts = [
                {"source_m_s": candidates[index].tolist(),
                 "coarse_total_loss": float(total[index])}
                for index in indices
            ]
            winner, _ = runner.refine(
                model, starts, domain, bounds, fine, picks, search,
                arrival_spec, alpha)
            branches[name] = winner
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        with np.load(path) as archive:
            truth = np.asarray(archive["source_truth"], dtype=np.float64)
        for branch in branches.values():
            error = np.asarray(branch["estimated_source_m_s"]) - truth
            branch["component_error_m_s"] = error.tolist()
            branch["location_error_m"] = float(np.linalg.norm(error[:3]))
            branch["origin_time_error_s"] = float(abs(error[3]))
        rows.append({
            "event": path.name,
            "sha256": runner.sha256(path),
            "valid_corrected_picks": int(np.isfinite(picks).sum()),
            "source_truth_m_s_post_search": truth.tolist(),
            "combined_neural_search_seconds": elapsed,
            **branches,
        })
        print(path.name + ": " + ", ".join(
            f"{name}={branch['location_error_m']:.1f} m"
            for name, branch in branches.items()), flush=True)

    result = {
        "purpose": "current-version neural replay of archived, inspected events",
        "independent_confirmation": False,
        "historical_byte_identical_reproduction": False,
        "limitations": "Installation check on already-inspected 5-Hz synthetic "
                       "records; the historical add81f runner is unavailable.",
        "device": str(device),
        "current_runner_sha256": runner.sha256(Path(runner.__file__)),
        "protocol_sha256": runner.sha256(PROTOCOL),
        "neural_projection_sha256": runner.sha256(PROJECTION),
        "checkpoint_sha256": spec["model_checkpoint_sha256"],
        "receiver_cache_sha256": spec["receiver_fmm_fields_sha256"],
        "event_count": len(rows),
        "summary": {name: runner.summarize(rows, name) for name in BRANCHES},
        "events": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Saved new replay output: " + str(output))


if __name__ == "__main__":
    main()
