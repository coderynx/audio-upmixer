"""Characterize the incumbent Roformer chunk schedule and batch invariance."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from upmixer.separation.inference import demix  # noqa: E402
from upmixer.separation.inference.config import ModelConfig  # noqa: E402


_CHUNK_SIZE = 16
_OVERLAPS = (1, 2, 4)
_BATCH_SIZES = (1, 2, 3)


def _make_config(n_targets: int) -> ModelConfig:
    instruments = ["vocals", "other"]
    target_instrument = "vocals"
    if n_targets > 1:
        instruments = [f"stem-{index}" for index in range(n_targets)]
        target_instrument = None
    return ModelConfig(
        audio={"sample_rate": 44100, "hop_length": 4},
        model={"stft_hop_length": 4},
        training={
            "instruments": instruments,
            "target_instrument": target_instrument,
        },
        inference={"dim_t": 5},
    )


def _make_mix(n_samples: int) -> np.ndarray:
    sample_numbers = np.arange(n_samples, dtype=np.float32)
    return np.stack((100.0 + sample_numbers, -50.0 + sample_numbers))


class _InputSensitiveModel(torch.nn.Module):
    """Stateless fake whose output depends only on each input window."""

    def __init__(self, n_targets: int) -> None:
        super().__init__()
        self._n_targets = n_targets

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self._n_targets == 1:
            return batch * 0.5 + 0.25
        scales = torch.arange(
            1, self._n_targets + 1, device=batch.device, dtype=batch.dtype
        ).view(1, self._n_targets, 1, 1)
        return batch.unsqueeze(1) * scales + scales * 0.25


class _RecordingModel:
    """Observe windows while leaving the fake model's outputs stateless."""

    def __init__(self, model: _InputSensitiveModel) -> None:
        self._model = model
        self.batches: list[torch.Tensor] = []
        self.forward_calls = 0
        self.evaluated_examples = 0

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.evaluated_examples += batch.shape[0]
        self.batches.append(batch.detach().cpu())
        return self._model(batch)


_EXPECTED_STARTS = {
    "below": {
        1: (0,),
        2: (0, 0),
        4: (0, 0, 0, 0),
    },
    "exact": {
        1: (0,),
        2: (0, 0),
        4: (0, 0, 0, 0),
    },
    "above": {
        1: (0, 1),
        2: (0, 1, 1),
        4: (0, 1, 1, 1, 1),
    },
    "nondiv_tail": {
        1: (0, 13),
        2: (0, 8, 13, 13),
        4: (0, 4, 8, 12, 13, 13, 13, 13),
    },
}

_CASE_LENGTHS = {
    "below": _CHUNK_SIZE - 1,
    "exact": _CHUNK_SIZE,
    "above": _CHUNK_SIZE + 1,
    "nondiv_tail": 29,
}


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
@pytest.mark.parametrize("case", _CASE_LENGTHS)
@pytest.mark.parametrize("overlap", _OVERLAPS)
def test_effective_windows_are_evaluated_once_but_tail_multiplicity_is_retained(
    batch_size: int, case: str, overlap: int
) -> None:
    recorded = _RecordingModel(_InputSensitiveModel(n_targets=1))
    demix.demix_roformer(
        recorded,
        _make_mix(_CASE_LENGTHS[case]),
        _make_config(n_targets=1),
        torch.device("cpu"),
        segment_size=None,
        overlap=overlap,
        batch_size=batch_size,
    )

    starts = tuple(
        int(sample.item() - 100)
        for batch in recorded.batches
        for sample in batch[:, 0, 0]
    )
    incumbent_starts = _EXPECTED_STARTS[case][overlap]
    unique_starts = tuple(dict.fromkeys(incumbent_starts))
    assert starts == unique_starts
    assert len(incumbent_starts) > 0
    assert recorded.evaluated_examples == len(unique_starts)
    assert recorded.forward_calls == (
        len(unique_starts) + batch_size - 1
    ) // batch_size


def _frozen_reference(
    model: _InputSensitiveModel,
    mix: np.ndarray,
    config: ModelConfig,
    overlap: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Run the pre-optimization schedule, including repeated tail positions."""
    mix_t = torch.tensor(mix, dtype=torch.float32)
    original_length = mix_t.shape[1]
    chunk_size = config.stft_hop_length * (config.default_segment_size - 1)
    if original_length < chunk_size:
        mix_t = torch.nn.functional.pad(mix_t, (0, chunk_size - original_length))
    n_samples = mix_t.shape[1]
    step = max(1, chunk_size // max(1, overlap))
    starts = [
        i if i + chunk_size <= n_samples else n_samples - chunk_size
        for i in range(0, n_samples, step)
    ]
    window = torch.tensor(np.hamming(chunk_size), dtype=torch.float32)
    num_stems = config.num_stems
    acc_shape = mix_t.shape if num_stems == 1 else (num_stems, *mix_t.shape)
    result = torch.zeros(acc_shape, dtype=torch.float32)
    counter = torch.zeros(acc_shape, dtype=torch.float32)
    for batch_start in range(0, len(starts), max(1, batch_size)):
        batch_starts = starts[batch_start : batch_start + max(1, batch_size)]
        batch = torch.stack([mix_t[:, s : s + chunk_size] for s in batch_starts])
        outputs = model(batch)
        for start, output in zip(batch_starts, outputs):
            result[..., start : start + chunk_size] += output * window
            counter[..., start : start + chunk_size] += window
    inferenced = (result / counter.clamp(min=1e-10)).numpy()
    if num_stems == 1:
        primary = inferenced[..., :original_length]
        secondary = next(
            name for name in config.instruments if name != config.target_instrument
        )
        return {
            config.target_instrument: primary,
            secondary: mix - primary,
        }
    trimmed = inferenced[..., :original_length]
    return dict(zip(config.instruments, trimmed))


@pytest.mark.parametrize("n_targets", (1, 2))
@pytest.mark.parametrize("case", _CASE_LENGTHS)
@pytest.mark.parametrize("overlap", _OVERLAPS)
def test_waveforms_are_invariant_to_batch_partition(
    n_targets: int, case: str, overlap: int
) -> None:
    mix = _make_mix(_CASE_LENGTHS[case])
    reference = _frozen_reference(
        _InputSensitiveModel(n_targets),
        mix,
        _make_config(n_targets),
        overlap,
        batch_size=1,
    )
    results = {
        batch_size: demix.demix_roformer(
            _InputSensitiveModel(n_targets),
            mix,
            _make_config(n_targets),
            torch.device("cpu"),
            segment_size=None,
            overlap=overlap,
            batch_size=batch_size,
        )
        for batch_size in _BATCH_SIZES
    }

    assert set(reference) == set(results[1]) == set(results[2]) == set(results[3])
    for name, expected in reference.items():
        np.testing.assert_allclose(
            results[1][name], expected, atol=1e-6, rtol=1e-5
        )
        assert expected.shape == (2, _CASE_LENGTHS[case])
        for batch_size in _BATCH_SIZES:
            np.testing.assert_allclose(
                results[batch_size][name],
                expected,
                atol=1e-6,
                rtol=1e-5,
            )
