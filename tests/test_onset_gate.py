"""Source-dependent onset gating and checkpoint persistence."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from source_conditioned_pinn import (SourceConditionedConfig,
                                     TravelTimeCarrierWavePINN,
                                     load_conditioned_checkpoint,
                                     save_conditioned_checkpoint)


GATE = {"maximum_speed_m_s": 6000.0, "support_m": 200.0,
        "lead_s": 0.20, "rise_s": 0.10}


def test_gate_has_zero_before_onset_and_finite_source_and_pde_gradients():
    model = TravelTimeCarrierWavePINN(5, 2, hidden_dim=8, num_layers=2,
                                      distance_core_m=20, onset_gate=GATE)
    # Coincident receiver/source starts gating at ts - lead; the middle
    # point lies inside the smooth transition instead of its flat tail.
    source = torch.tensor([[0.5, 0.5, 0.3, 0.125]], requires_grad=True)
    points = torch.tensor([[0.5, 0.5, 0.3, 0.01],
                           [0.5, 0.5, 0.3, 0.05]], requires_grad=True)
    gated = model._apply_onset_gate(torch.ones(2, 1), points,
                                    source.expand(2, 4))
    assert gated[0].item() == 0
    assert abs(gated[1].item() - 0.5) < 1e-6
    source_grad = torch.autograd.grad(gated[1], source, retain_graph=True)[0]
    assert torch.isfinite(source_grad).all()
    assert source_grad[0, 3].abs().item() > 0
    time_grad = torch.autograd.grad(gated[1], points, create_graph=True)[0]
    curvature = torch.autograd.grad(time_grad[1, 3], points)[0]
    assert torch.isfinite(curvature).all()


def test_gate_depends_on_source_position_away_from_source_support():
    model = TravelTimeCarrierWavePINN(5, 2, hidden_dim=8, num_layers=2,
                                      distance_core_m=20, onset_gate=GATE)
    source = torch.tensor([[0.5, 0.5, 0.3, 0.125]], requires_grad=True)
    # 1600 m offset places the gate transition after the spatial support.
    point = torch.tensor([[0.6, 0.5, 0.3, 0.1665]], requires_grad=True)
    gated = model._apply_onset_gate(torch.ones(1, 1), point, source)
    assert 0 < gated.item() < 1
    gradient = torch.autograd.grad(gated.sum(), source)[0]
    assert gradient[0, 0].abs().item() > 0
    assert gradient[0, 3].abs().item() > 0


def test_gate_checkpoint_round_trip_and_ungated_default():
    model = TravelTimeCarrierWavePINN(5, 2, hidden_dim=8, num_layers=2,
                                      distance_core_m=20, onset_gate=GATE)
    points = torch.tensor([[0.5, 0.6, 0.1, 0.5]])
    sources = torch.tensor([[0.5, 0.4, 0.1, 0.1]])
    with TemporaryDirectory() as directory:
        path = Path(directory) / "gated.pth"
        save_conditioned_checkpoint(path, model, None, 5, 1,
                                    SourceConditionedConfig(time_length_s=2))
        restored, payload = load_conditioned_checkpoint(path, torch.device("cpu"))
        assert payload["architecture"]["onset_gate"] == GATE
        torch.testing.assert_close(restored(points, sources), model(points, sources))
    assert TravelTimeCarrierWavePINN(5, 2).onset_gate is None
