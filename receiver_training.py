"""Receiver supervision from explicitly named FD training events only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from source_conditioned_pinn import SourceConditionedConfig


class ReceiverTrainingData:
    def __init__(self, paths: list[Path], device: torch.device,
                 config: SourceConditionedConfig, frequency_hz: float,
                 seed: int = 42):
        if not paths:
            raise ValueError("At least one training observation is required")
        self._numpy_rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator(device=device)
        self._torch_rng.manual_seed(int(seed))
        scale = np.array([config.x_length_m, config.y_length_m,
                          config.z_length_m, config.time_length_s], dtype=np.float32)
        self.events = []
        self.metadata = []
        sources = set()
        reference_receivers = None
        reference_times = None
        for path in paths:
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            report_path = path.with_suffix(".json")
            if not report_path.is_file():
                raise ValueError(f"Missing FD provenance report: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("method") != "second_order_3d_acoustic_FD"
                    or report.get("frequency_hz") != float(frequency_hz)):
                raise ValueError(f"FD solver or frequency does not match: {path}")
            with np.load(path) as archive:
                data = np.asarray(archive["data"], dtype=np.float32)
                times = np.asarray(archive["times"], dtype=np.float32)
                receivers = np.asarray(archive["receiver_coords"], dtype=np.float32)
                source = np.asarray(archive["source_truth"], dtype=np.float32)
            if data.shape != (len(receivers), len(times)) or source.shape != (4,):
                raise ValueError(f"Bad FD training event dimensions: {path}")
            if not (np.isfinite(data).all() and np.isfinite(times).all()
                    and np.isfinite(receivers).all() and np.isfinite(source).all()):
                raise ValueError(f"Nonfinite training event: {path}")
            if np.any(receivers < 0) or np.any(receivers > scale[:3]) or np.any(source < 0):
                raise ValueError(f"Coordinates outside model: {path}")
            if times.min() < 0 or times.max() > config.time_length_s + 1e-5:
                raise ValueError(f"Time outside training domain: {path}")
            normalized_source = source / scale
            lower = np.array([config.source_x_min, config.source_y_min,
                              config.source_z_min, config.source_time_min])
            upper = np.array([config.source_x_max, config.source_y_max,
                              config.source_z_max, config.source_time_max])
            if np.any(normalized_source < lower - 1e-6) or np.any(normalized_source > upper + 1e-6):
                raise ValueError(f"Training source outside conditioned bounds: {path}")
            if reference_receivers is None:
                reference_receivers, reference_times = receivers, times
            elif not (np.array_equal(receivers, reference_receivers)
                      and np.array_equal(times, reference_times)):
                raise ValueError(f"Receiver geometry or time axis differs: {path}")
            key = tuple(np.round(source, 4).tolist())
            if key in sources:
                raise ValueError(f"Duplicate training source: {source.tolist()}")
            sources.add(key)
            rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
            peak = float(np.max(np.abs(data)))
            if rms <= 0 or peak <= 0:
                raise ValueError(f"Empty waveform: {path}")
            n_receiver, n_time = data.shape
            coords = np.concatenate((
                np.broadcast_to(receivers[:, None, :], (n_receiver, n_time, 3)),
                np.broadcast_to(times[None, :, None], (n_receiver, n_time, 1)),
            ), axis=-1).reshape(-1, 4) / scale
            values = data.ravel() / rms
            active = np.flatnonzero(np.abs(data.ravel()) >= 0.05 * peak)
            if len(active) == 0:
                raise ValueError(f"No active samples: {path}")
            self.events.append((torch.as_tensor(coords, device=device),
                                torch.as_tensor(values, device=device),
                                torch.as_tensor(source / scale, device=device),
                                torch.as_tensor(active, device=device), rms))
            self.metadata.append({"path": str(path),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "source_truth": source.tolist(),
                                  "frequency_hz": frequency_hz, "rms": rms,
                                  "active_sample_count": len(active)})

    def _sample_event(self, event, count: int):
        x, y, src, active, rms = event
        half = count // 2
        active_indices = active[torch.randint(
            len(active), (half,), device=x.device, generator=self._torch_rng)]
        uniform_indices = torch.randint(
            len(x), (count - half,), device=x.device, generator=self._torch_rng)
        index = torch.cat((active_indices, uniform_indices))
        return (x[index], y[index], src.expand(count, 4),
                torch.full((count,), rms, device=x.device))

    def batch(self, batch_size: int, event_count: int = 1):
        if batch_size < 2:
            raise ValueError("Receiver batch size must be at least 2")
        if event_count < 1 or event_count > len(self.events):
            raise ValueError("Receiver event count must be within the training set")
        if batch_size < 2 * event_count:
            raise ValueError("Receiver batch needs at least two samples per event")
        selected = self._numpy_rng.choice(len(self.events), event_count, replace=False)
        base, remainder = divmod(batch_size, event_count)
        pieces = [self._sample_event(self.events[index],
                                     base + (position < remainder))
                  for position, index in enumerate(selected)]
        return tuple(torch.cat(items, dim=0) for items in zip(*pieces))
