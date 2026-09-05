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

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
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


@pytest.mark.parametrize("case", _CASE_LENGTHS)
@pytest.mark.parametrize("overlap", _OVERLAPS)
def test_incumbent_start_schedule_includes_clamped_tail_repeats(
    case: str, overlap: int
) -> None:
    recorded = _RecordingModel(_InputSensitiveModel(n_targets=1))
    demix.demix_roformer(
        recorded,
        _make_mix(_CASE_LENGTHS[case]),
        _make_config(n_targets=1),
        torch.device("cpu"),
        segment_size=None,
        overlap=overlap,
        batch_size=3,
    )

    starts = tuple(
        int(sample.item() - 100)
        for batch in recorded.batches
        for sample in batch[:, 0, 0]
    )
    assert starts == _EXPECTED_STARTS[case][overlap]


@pytest.mark.parametrize("n_targets", (1, 2))
@pytest.mark.parametrize("case", _CASE_LENGTHS)
@pytest.mark.parametrize("overlap", _OVERLAPS)
def test_waveforms_are_invariant_to_batch_partition(
    n_targets: int, case: str, overlap: int
) -> None:
    mix = _make_mix(_CASE_LENGTHS[case])
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

    reference = results[1]
    assert set(reference) == set(results[2]) == set(results[3])
    for name, expected in reference.items():
        assert expected.shape == (2, _CASE_LENGTHS[case])
        for batch_size in _BATCH_SIZES[1:]:
            np.testing.assert_allclose(
                results[batch_size][name],
                expected,
                atol=1e-6,
                rtol=1e-5,
            )
