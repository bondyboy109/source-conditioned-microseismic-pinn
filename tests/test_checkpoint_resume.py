"""Compatibility checks for receiver-pretrained physics fine-tuning."""

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from source_conditioned_pinn import (
    SourceConditionedConfig,
    TranslationInvariantTravelTimeCarrierWavePINN,
    load_conditioned_checkpoint,
    save_conditioned_checkpoint,
)
from train_source_conditioned_pinn import validate_initial_checkpoint


def arguments(**overrides):
    values = dict(
        frequencies=[5.0], translation_invariant_carrier=True,
        travel_time_carrier=False, harmonic_carrier=False,
        wavefield_scale=1e-5, hard_initial=False,
        reference_velocity=3500.0, distance_core_m=1.0,
        ray_travel_time=False, ray_samples=16,
        eikonal_checkpoint=None,
    )
    values.update(overrides)
    return Namespace(**values)


class CheckpointResumeTests(unittest.TestCase):
    def make_checkpoint(self, directory: str):
        cfg = SourceConditionedConfig()
        model = TranslationInvariantTravelTimeCarrierWavePINN(
            5.0, cfg.time_length_s, reference_velocity_m_s=3500.0,
            x_length_m=cfg.x_length_m, y_length_m=cfg.y_length_m,
            z_length_m=cfg.z_length_m, output_scale=1e-5,
        )
        path = Path(directory) / "initial.pth"
        save_conditioned_checkpoint(path, model, None, 5.0, 10, cfg)
        return path, cfg

    def test_compatible_checkpoint_can_initialize_fine_tuning(self):
        with TemporaryDirectory() as directory:
            path, cfg = self.make_checkpoint(directory)
            _, payload = load_conditioned_checkpoint(path, torch.device("cpu"))
            validate_initial_checkpoint(payload, arguments(), cfg)

    def test_architecture_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            path, cfg = self.make_checkpoint(directory)
            _, payload = load_conditioned_checkpoint(path, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_initial_checkpoint(
                    payload,
                    arguments(translation_invariant_carrier=False), cfg,
                )

    def test_reference_velocity_override_must_be_explicit(self):
        with TemporaryDirectory() as directory:
            path, cfg = self.make_checkpoint(directory)
            _, payload = load_conditioned_checkpoint(path, torch.device("cpu"))
            with self.assertRaisesRegex(ValueError, "reference velocity"):
                validate_initial_checkpoint(payload, arguments(reference_velocity=4500), cfg)
            validate_initial_checkpoint(
                payload,
                arguments(reference_velocity=4500,
                          allow_reference_velocity_mismatch=True), cfg,
            )


if __name__ == "__main__":
    unittest.main()
