"""Evaluation-only aggregate-Drums event/repetition repair (Q50)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from upmixer.analysis.stft import STFTAnalyzer
from upmixer.config import UpmixConfig

Q50_RECIPE = "q50-aggregate-drums-event-repetition-v1"
PRIMARY_SIBLINGS = ("Bass", "Drums", "Guitar", "Piano", "Other")
_FFT_SIZE = 1024
_HOP_LENGTH = 256
_PRE_FRAMES = 2
_POST_FRAMES = 10
_ATTACK_FRAMES = 5
_MAX_EVENTS = 48
_MAX_TRANSFERS = 8
_TRANSFER_GAIN = 0.25
_MIN_SPECTRAL_SIMILARITY = 0.99
_MIN_ATTACK_SIMILARITY = 0.98
_MIN_ONSET_SIMILARITY = 0.97
_MIN_STEREO_SIMILARITY = 0.97
_MIN_PARENT_SPECTRAL_SIMILARITY = 0.80
_MIN_PARENT_ONSET_SIMILARITY = 0.90
_MIN_PARENT_SUPPORT = 0.65
_AMBIGUITY_MARGIN = 0.03
_EPSILON = 1e-12


@dataclass(frozen=True)
class EventTransfer:
    """One accepted donor-to-Drums transfer."""

    donor: str
    donor_frame: int
    reference_frame: int
    sample_start: int
    sample_end: int
    gain: float
    score: float


@dataclass(frozen=True)
class AggregateDrumsRepairResult:
    """Repaired sibling estimates and the transfers accepted by Q50."""

    stems: dict[str, np.ndarray]
    transfers: tuple[EventTransfer, ...]
    recipe: str = Q50_RECIPE


@dataclass(frozen=True)
class _Event:
    frame: int
    energy: float
    spectrum: np.ndarray
    profile: np.ndarray
    attack: np.ndarray
    stereo: np.ndarray


@dataclass(frozen=True)
class _Match:
    donor: str
    donor_event: _Event
    reference_event: _Event
    score: float
    start: int
    end: int


def _validate_audio(audio: object, label: str) -> np.ndarray:
    if not isinstance(audio, np.ndarray) or audio.ndim != 2 or not audio.size:
        raise ValueError(f"{label} must be a non-empty 2D array")
    if audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError(f"{label} must have positive frame and channel counts")
    if audio.dtype.kind != "f" or not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} must contain finite floating-point values")
    return audio


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _best_local_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Allow one STFT-frame of onset quantization while matching events."""
    scores = [_cosine(left, right)]
    for shift in (-1, 1):
        if shift < 0:
            scores.append(_cosine(left[:shift], right[-shift:]))
        else:
            scores.append(_cosine(left[shift:], right[:-shift]))
    return max(scores)


def _stereo_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Keep channel balance and inter-channel phase close, not just aligned."""
    return max(0.0, 1.0 - 2.0 * float(np.mean(np.abs(left - right))))


def _fixed_slice(values: np.ndarray, center: int, length: int) -> np.ndarray:
    """Return a zero-padded, fixed-length frame vector around ``center``."""
    half = _PRE_FRAMES
    start = center - half
    result = np.zeros(length, dtype=np.float64)
    source_start = max(0, start)
    source_end = min(values.shape[0], start + length)
    if source_end > source_start:
        result[source_start - start : source_end - start] = values[
            source_start:source_end
        ]
    return result


def _stereo_descriptor(audio: np.ndarray, frame: int) -> np.ndarray:
    center = frame * _HOP_LENGTH
    start = max(0, center - _PRE_FRAMES * _HOP_LENGTH)
    end = min(audio.shape[0], center + _POST_FRAMES * _HOP_LENGTH)
    window = np.asarray(audio[start:end], dtype=np.float64)
    if window.shape[0] < 2:
        return np.zeros(3, dtype=np.float64)
    channel_rms = np.sqrt(np.mean(window**2, axis=0))
    total = float(np.sum(channel_rms))
    if total <= _EPSILON:
        balance = np.zeros(2, dtype=np.float64)
    else:
        balance = channel_rms[:2] / total
    centered = window[:, :2] - np.mean(window[:, :2], axis=0)
    denominator = float(np.linalg.norm(centered[:, 0]) * np.linalg.norm(centered[:, 1]))
    correlation = (
        float(np.dot(centered[:, 0], centered[:, 1]) / denominator)
        if denominator > _EPSILON
        else 0.0
    )
    return np.array((balance[0], balance[1], (correlation + 1.0) * 0.5))


def _event_frames(energy: np.ndarray, sample_rate: int) -> list[int]:
    if energy.size < 3:
        return []
    maximum = float(np.max(energy))
    if not np.isfinite(maximum) or maximum <= _EPSILON:
        return []
    onset = np.maximum(energy - np.r_[0.0, energy[:-1]], 0.0)
    positive = onset[onset > _EPSILON]
    if not positive.size:
        return []
    threshold = max(float(np.quantile(positive, 0.75)), float(np.max(onset)) * 0.10)
    candidates = [
        frame
        for frame in range(1, onset.shape[0] - 1)
        if onset[frame] >= threshold
        and onset[frame] >= onset[frame - 1]
        and onset[frame] >= onset[frame + 1]
        and energy[frame] >= maximum * 0.01
    ]
    # A fixed time gap prevents one attack's STFT skirt becoming many events.
    min_gap = max(4, int(0.06 * sample_rate / _HOP_LENGTH))
    selected: list[int] = []
    for frame in sorted(candidates, key=lambda value: (-onset[value], value)):
        if all(abs(frame - other) >= min_gap for other in selected):
            selected.append(frame)
        if len(selected) == _MAX_EVENTS:
            break
    return sorted(selected)


def _events(
    audio: np.ndarray,
    sample_rate: int,
    extra_frames: tuple[int, ...] = (),
) -> list[_Event]:
    analyzer = STFTAnalyzer(
        UpmixConfig(
            fft_size=_FFT_SIZE,
            hop_size=_HOP_LENGTH,
            auto_fft_size=False,
        ),
        sample_rate,
    )
    spectra = np.stack([analyzer.forward(audio[:, channel]) for channel in range(2)])
    magnitude = np.abs(spectra).astype(np.float64, copy=False)
    energy = np.sqrt(np.mean(magnitude[:, 1:] ** 2, axis=(0, 1)))
    result: list[_Event] = []
    frames = sorted(
        set(_event_frames(energy, sample_rate)).union(
            frame for frame in extra_frames if 0 <= frame < energy.shape[0]
        )
    )
    for frame in frames:
        start = max(0, frame - _PRE_FRAMES)
        end = min(magnitude.shape[2], frame + _POST_FRAMES + 1)
        spectrum = np.mean(magnitude[:, 1:, start:end], axis=(0, 2))
        spectrum /= max(float(np.linalg.norm(spectrum)), _EPSILON)
        profile = _fixed_slice(energy, frame, _PRE_FRAMES + _POST_FRAMES + 1)
        profile /= max(float(np.linalg.norm(profile)), _EPSILON)
        attack_start = _PRE_FRAMES
        attack = profile[attack_start : attack_start + _ATTACK_FRAMES].copy()
        attack /= max(float(np.linalg.norm(attack)), _EPSILON)
        result.append(
            _Event(
                frame=frame,
                energy=float(energy[frame]),
                spectrum=spectrum,
                profile=profile,
                attack=attack,
                stereo=_stereo_descriptor(audio, frame),
            )
        )
    return result


def _nearest_event(events: list[_Event], frame: int) -> _Event | None:
    if not events:
        return None
    event = min(
        events, key=lambda candidate: (abs(candidate.frame - frame), candidate.frame)
    )
    return event if abs(event.frame - frame) <= 2 else None


def _event_window(frame: int, frame_count: int) -> tuple[int, int]:
    center = frame * _HOP_LENGTH
    return (
        max(0, center - _PRE_FRAMES * _HOP_LENGTH),
        min(frame_count, center + (_POST_FRAMES + 1) * _HOP_LENGTH),
    )


def _similarity(
    donor: _Event,
    reference: _Event,
    parent_donor: _Event,
    parent_reference: _Event,
) -> float | None:
    spectral = _cosine(donor.spectrum, reference.spectrum)
    attack = _best_local_cosine(donor.attack, reference.attack)
    onset = _best_local_cosine(donor.profile, reference.profile)
    stereo = _stereo_agreement(donor.stereo, reference.stereo)
    parent_spectral = min(
        _cosine(parent_donor.spectrum, donor.spectrum),
        _cosine(parent_reference.spectrum, reference.spectrum),
    )
    parent_onset = min(
        _best_local_cosine(parent_donor.profile, donor.profile),
        _best_local_cosine(parent_reference.profile, reference.profile),
    )
    parent_support = min(
        parent_donor.energy / max(donor.energy, _EPSILON),
        parent_reference.energy / max(reference.energy, _EPSILON),
    )
    if (
        spectral < _MIN_SPECTRAL_SIMILARITY
        or attack < _MIN_ATTACK_SIMILARITY
        or onset < _MIN_ONSET_SIMILARITY
        or stereo < _MIN_STEREO_SIMILARITY
        or parent_spectral < _MIN_PARENT_SPECTRAL_SIMILARITY
        or parent_onset < _MIN_PARENT_ONSET_SIMILARITY
        or parent_support < _MIN_PARENT_SUPPORT
    ):
        return None
    return float(
        0.30 * spectral
        + 0.20 * attack
        + 0.15 * onset
        + 0.15 * stereo
        + 0.10 * parent_spectral
        + 0.10 * parent_onset
    )


def _validate_inputs(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    _validate_audio(parent, "parent")
    if not isinstance(stems, Mapping) or "Drums" not in stems:
        raise ValueError("stems must contain Drums")
    checked: dict[str, np.ndarray] = {}
    for name, audio in stems.items():
        if not isinstance(name, str):
            raise ValueError("stem names must be strings")
        checked[name] = _validate_audio(audio, f"stem {name}")
        if checked[name].shape != parent.shape:
            raise ValueError(f"stem {name} shape does not match parent")
    return checked


def repair_aggregate_drums(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
    sample_rate: int = 44_100,
) -> AggregateDrumsRepairResult:
    """Apply fixed Q50 event transfers to primary sibling estimates.

    A donor event is eligible only when a separated Drums event elsewhere in the
    same track agrees in spectrum, attack, onset profile, stereo image, and
    parent support.  The transferred waveform is the donor's current signal;
    the retrieved event supplies descriptors only.
    """
    checked = _validate_inputs(parent, stems)
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate < 1
    ):
        raise ValueError("sample_rate must be a positive integer")
    output = {name: audio.copy() for name, audio in checked.items()}
    if parent.shape[1] != 2 or not any(np.any(audio) for audio in checked.values()):
        return AggregateDrumsRepairResult(output, ())

    drum_events = _events(checked["Drums"], sample_rate)
    donor_event_sets = {
        donor_name: _events(checked[donor_name], sample_rate)
        for donor_name in PRIMARY_SIBLINGS
        if donor_name != "Drums" and donor_name in checked
    }
    donor_frames = tuple(
        event.frame for events in donor_event_sets.values() for event in events
    )
    parent_events = _events(
        parent,
        sample_rate,
        donor_frames + tuple(event.frame for event in drum_events),
    )
    if len(drum_events) < 2 or not parent_events:
        return AggregateDrumsRepairResult(output, ())

    frame_count = parent.shape[0]
    matches: list[_Match] = []
    min_gap = max(4, int(0.06 * sample_rate / _HOP_LENGTH))
    for donor_name in PRIMARY_SIBLINGS:
        if donor_name == "Drums" or donor_name not in checked:
            continue
        donor_events = donor_event_sets[donor_name]
        for donor_event in donor_events:
            parent_donor = _nearest_event(parent_events, donor_event.frame)
            if parent_donor is None:
                continue
            candidates: list[_Match] = []
            for reference_event in drum_events:
                if abs(donor_event.frame - reference_event.frame) < min_gap:
                    continue
                parent_reference = _nearest_event(parent_events, reference_event.frame)
                if parent_reference is None:
                    continue
                score = _similarity(
                    donor_event,
                    reference_event,
                    parent_donor,
                    parent_reference,
                )
                if score is not None:
                    start, end = _event_window(donor_event.frame, frame_count)
                    candidates.append(
                        _Match(
                            donor=donor_name,
                            donor_event=donor_event,
                            reference_event=reference_event,
                            score=score,
                            start=start,
                            end=end,
                        )
                    )
            candidates.sort(
                key=lambda match: (
                    -match.score,
                    match.reference_event.frame,
                )
            )
            if candidates and (
                len(candidates) == 1
                or candidates[0].score - candidates[1].score >= _AMBIGUITY_MARGIN
            ):
                matches.append(candidates[0])

    # Resolve equal-strength donor candidates deterministically by abstaining.
    matches.sort(key=lambda match: (-match.score, match.donor, match.donor_event.frame))
    accepted: list[_Match] = []
    for match in matches:
        conflict = next(
            (
                other
                for other in accepted
                if not (match.end <= other.start or match.start >= other.end)
            ),
            None,
        )
        if conflict is not None:
            if abs(match.score - conflict.score) < _AMBIGUITY_MARGIN:
                accepted.remove(conflict)
            continue
        if len(accepted) >= _MAX_TRANSFERS:
            break
        accepted.append(match)

    transfers: list[EventTransfer] = []
    for match in sorted(accepted, key=lambda value: (value.start, value.donor)):
        source = output[match.donor][match.start : match.end]
        available = output["Drums"][match.start : match.end]
        residual = np.asarray(
            parent[match.start : match.end], dtype=np.float64
        ) - np.asarray(available, dtype=np.float64)
        source_rms = float(np.sqrt(np.mean(np.asarray(source, dtype=np.float64) ** 2)))
        available_rms = float(np.sqrt(np.mean(residual**2)))
        gain = min(_TRANSFER_GAIN, 0.5 * available_rms / max(source_rms, _EPSILON))
        if gain <= _EPSILON:
            continue
        length = match.end - match.start
        fade = min(_HOP_LENGTH // 2, max(1, length // 8))
        mask = np.ones(length, dtype=np.float64)
        mask[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        mask[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=False)
        delta = np.asarray(source, dtype=np.float64) * (mask[:, None] * gain)
        with np.errstate(over="ignore", invalid="ignore"):
            drum_update = np.asarray(available, dtype=np.float64) + delta
            donor_update = np.asarray(source, dtype=np.float64) - delta
        if not np.all(np.isfinite(drum_update)) or not np.all(
            np.isfinite(donor_update)
        ):
            continue
        output["Drums"][match.start : match.end] = drum_update.astype(
            output["Drums"].dtype
        )
        output[match.donor][match.start : match.end] = donor_update.astype(
            output[match.donor].dtype
        )
        transfers.append(
            EventTransfer(
                donor=match.donor,
                donor_frame=match.donor_event.frame,
                reference_frame=match.reference_event.frame,
                sample_start=match.start,
                sample_end=match.end,
                gain=float(gain),
                score=float(match.score),
            )
        )
    if not all(np.all(np.isfinite(audio)) for audio in output.values()):
        return AggregateDrumsRepairResult(
            {name: audio.copy() for name, audio in checked.items()},
            (),
        )
    return AggregateDrumsRepairResult(output, tuple(transfers))


__all__ = [
    "Q50_RECIPE",
    "PRIMARY_SIBLINGS",
    "EventTransfer",
    "AggregateDrumsRepairResult",
    "repair_aggregate_drums",
]
