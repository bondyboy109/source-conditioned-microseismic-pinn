import json

from examples.measure_compute_cost import collect


def test_break_even_direction(tmp_path):
    summary = {
        "run": {
            "device": "cuda", "torch_version": "test", "numpy_version": "test",
            "seed": 1, "parameter_count": 10,
            "receiver_observations": [{}, {}],
        },
        "5Hz": {"seconds": 20.0, "peak_cuda_memory_bytes": 2**20},
    }
    comparison = {
        "pinn_setup_seconds": 0.0,
        "fmm_setup_seconds": 30.0,
        "fmm_receiver_fields_mib": 4.0,
        "events": [
            {"pinn": {"runtime_seconds": 4.0},
             "fmm": {"status": "ok", "search_seconds": 1.0},
             "fd_runtime_s": 10.0},
            {"pinn": {"runtime_seconds": 4.0},
             "fmm": {"status": "ok", "search_seconds": 1.0},
             "fd_runtime_s": 10.0},
        ],
    }
    summary_path = tmp_path / "summary.json"
    comparison_path = tmp_path / "comparison.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    result = collect([summary_path], comparison_path)
    crossovers = result["break_even_events"]

    assert crossovers["pinn_vs_fd_inference_only"] == 4
    assert crossovers["fmm_vs_fd_inference_only"] == 4
    assert crossovers["pinn_per_event_faster_than_fmm"] is False
    assert crossovers["fmm_becomes_cheaper_than_pinn_at_event"] == 4
    assert crossovers["pinn_vs_fd_including_training_record_generation"] == 7
