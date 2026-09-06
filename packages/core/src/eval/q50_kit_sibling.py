"""Bounded, evaluation-only Q50 kit-sibling transfer arms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np

Q50_KIT_RECIPE = "q50-kit-sibling-v2"
KIT_OUTPUTS = ("Kick", "Snare", "Toms", "Hi-Hat", "Ride", "Crash")
KIT_TARGETS = ("Kick", "Snare", "Hi-Hat")
KIT_ARMS = ("cue-only", "cue-plus-repetition")

_BLOCK_SECONDS = 2.0
_BLOCK_OVERLAP_SECONDS = 0.125
_MAX_DESCRIPTORS = 96
_MAX_PROPOSALS_PER_BLOCK = 2
_MAX_PROPOSALS = 16
_WINDOW_SECONDS = 0.16
_PRE_SECONDS = 0.015
_ATTACK_SECONDS = 0.04
_MIN_EVENT_SECONDS = 0.04
_MAX_GAIN = 0.15
_MIN_SPECTRAL = 0.88
_MIN_ENVELOPE = 0.88
_MIN_ATTACK = 0.86
_MIN_DECAY = 0.78
_MIN_STEREO = 0.90
_MIN_COHERENCE = 0.35
_MIN_DONOR_RATIO = 0.40
_MIN_PARENT_RATIO = 0.25
_MAX_TARGET_RATIO = 0.12
_WINNER_MARGIN = 0.10
_EPSILON = 1e-12


@dataclass(frozen=True)
class KitSiblingProposal:
    """Immutable decision made from the original sibling arrays."""

    donor: str
    target: str
    sample_start: int
    sample_end: int
    gain: float
    score: float
    signed_coherence: float
    exemplar_count: int
    block_index: int


@dataclass(frozen=True)
class KitSiblingRepairResult:
    """One fixed Q50 arm's output and auditable transfer decisions."""

    stems: dict[str, np.ndarray]
    proposals: tuple[KitSiblingProposal, ...]
    arm: str
    descriptor_counts: tuple[tuple[str, int], ...]
    conservation_error: float
    recipe: str = Q50_KIT_RECIPE

    @property
    def transfer_count(self) -> int:
        return len(self.proposals)

    @property
    def repetition_transfer_count(self) -> int:
        return sum(proposal.exemplar_count >= 2 for proposal in self.proposals)


@dataclass(frozen=True)
class _Event:
    center: int
    start: int
    end: int
    rms: float
    bands: np.ndarray
    envelope: np.ndarray
    attack: np.ndarray
    decay: np.ndarray
    complex_attack: np.ndarray
    stereo: np.ndarray
    percussive: float


def _validate_audio(audio: object, label: str) -> np.ndarray:
    if not isinstance(audio, np.ndarray) or audio.ndim != 2 or not audio.size:
        raise ValueError(f"{label} must be a non-empty 2-D array")
    if audio.shape[0] < 1 or audio.shape[1] not in (1, 2):
        raise ValueError(f"{label} must have one or two channels")
    if audio.dtype.kind != "f" or not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} must contain finite floating-point values")
    return audio


def _validate_inputs(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
    sample_rate: int,
    arm: str,
) -> dict[str, np.ndarray]:
    _validate_audio(parent, "parent")
    if not isinstance(stems, Mapping):
        raise ValueError("stems must be a mapping")
    missing = [name for name in KIT_OUTPUTS if name not in stems]
    if missing:
        raise ValueError(f"stems missing DrumSep outputs: {', '.join(missing)}")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate < 1
    ):
        raise ValueError("sample_rate must be a positive integer")
    normalized_arm = arm.replace("_", "-") if isinstance(arm, str) else arm
    if normalized_arm not in KIT_ARMS:
        raise ValueError(f"arm must be one of {KIT_ARMS}")
    checked: dict[str, np.ndarray] = {}
    for name, audio in stems.items():
        if not isinstance(name, str):
            raise ValueError("stem names must be strings")
        value = _validate_audio(audio, f"stem {name}")
        if value.shape != parent.shape:
            raise ValueError(f"stem {name} shape does not match parent")
        checked[name] = value
    return checked


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _signed_coherence(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return 0.0
    return float(np.real(np.vdot(left, right)) / (left_norm * right_norm))


def _block_ranges(length: int, sample_rate: int) -> list[tuple[int, int, int]]:
    block = max(1, round(_BLOCK_SECONDS * sample_rate))
    overlap = min(block - 1, round(_BLOCK_OVERLAP_SECONDS * sample_rate))
    stride = max(1, block - overlap)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < length:
        end = min(length, start + block)
        ranges.append((index, start, end))
        if end == length:
            break
        start += stride
        index += 1
    return ranges


def _frame_features(
    signal: np.ndarray, frame_length: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    if signal.size < frame_length:
        padded = np.pad(signal, (0, frame_length - signal.size))
    else:
        padded = signal
    starts = range(0, max(1, len(padded) - frame_length + 1), hop)
    energies: list[float] = []
    spectra: list[np.ndarray] = []
    window = np.hanning(frame_length)
    for start in starts:
        frame = padded[start : start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        energies.append(float(np.sqrt(np.mean(frame * frame))))
        spectra.append(np.abs(np.fft.rfft(frame * window)))
    return np.asarray(energies), np.asarray(spectra)


def _event_centers(signal: np.ndarray, sample_rate: int) -> list[int]:
    frame_length = max(16, round(sample_rate * 0.01))
    hop = max(8, frame_length // 2)
    energies, spectra = _frame_features(signal, frame_length, hop)
    if not energies.size or float(np.max(energies)) <= _EPSILON:
        return []
    flux = np.zeros(len(spectra), dtype=np.float64)
    if len(spectra) > 1:
        previous = spectra[:-1] / np.maximum(
            np.linalg.norm(spectra[:-1], axis=1, keepdims=True), _EPSILON
        )
        current = spectra[1:] / np.maximum(
            np.linalg.norm(spectra[1:], axis=1, keepdims=True), _EPSILON
        )
        flux[1:] = np.sqrt(np.sum(np.maximum(current - previous, 0.0) ** 2, axis=1))
    onset = np.maximum(np.diff(energies, prepend=energies[0]), 0.0) + flux * max(
        float(np.max(energies)), _EPSILON
    )
    positive = onset[onset > _EPSILON]
    if not positive.size:
        return []
    floor = max(float(np.quantile(positive, 0.65)), float(np.max(onset)) * 0.08)
    noise = float(np.quantile(energies, 0.20))
    active_floor = max(noise * 4.0, float(np.max(energies)) * 0.01, _EPSILON)
    min_gap = max(round(sample_rate * _MIN_EVENT_SECONDS), frame_length * 2)
    candidates = [
        index
        for index in range(1, len(onset) - 1)
        if onset[index] >= floor
        and onset[index] >= onset[index - 1]
        and onset[index] >= onset[index + 1]
        and energies[index] >= active_floor
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda value: (-onset[value], value)):
        center = index * hop + frame_length // 2
        if all(abs(center - other) >= min_gap for other in selected):
            selected.append(center)
        if len(selected) >= _MAX_DESCRIPTORS:
            break
    return sorted(selected)


def _describe(audio: np.ndarray, center: int, sample_rate: int) -> _Event | None:
    length = max(16, round(sample_rate * _WINDOW_SECONDS))
    pre = min(length - 1, round(sample_rate * _PRE_SECONDS))
    start = max(0, center - pre)
    end = min(len(audio), start + length)
    start = max(0, end - length)
    window = np.asarray(audio[start:end], dtype=np.float64)
    if len(window) < length:
        window = np.pad(window, ((0, length - len(window)), (0, 0)))
    mono = np.mean(window[:, :2], axis=1)
    rms = float(np.sqrt(np.mean(mono * mono)))
    if rms <= _EPSILON:
        return None
    bins = np.array_split(mono, 16)
    envelope = np.asarray([np.sqrt(np.mean(part * part)) for part in bins])
    envelope /= max(float(np.linalg.norm(envelope)), _EPSILON)
    attack_bins = envelope[:5].copy()
    decay_bins = envelope[6:].copy()
    attack_bins /= max(float(np.linalg.norm(attack_bins)), _EPSILON)
    decay_bins /= max(float(np.linalg.norm(decay_bins)), _EPSILON)
    attack_len = min(len(mono), max(16, round(sample_rate * _ATTACK_SECONDS)))
    attack_signal = mono[:attack_len]
    n_fft = 1 << max(6, (attack_len - 1).bit_length())
    complex_attack = np.fft.rfft(attack_signal * np.hanning(attack_len), n=n_fft)
    magnitude = np.abs(complex_attack)[1:]
    band_parts = np.array_split(magnitude, 8)
    bands = np.asarray([float(np.mean(part)) for part in band_parts])
    bands /= max(float(np.linalg.norm(bands)), _EPSILON)
    channels = window[:, :2]
    channel_rms = np.sqrt(np.mean(channels * channels, axis=0))
    if channels.shape[1] == 1:
        stereo = np.asarray((1.0, 0.0, 1.0))
    else:
        total = float(np.sum(channel_rms))
        centered = channels - np.mean(channels, axis=0)
        denominator = float(
            np.linalg.norm(centered[:, 0]) * np.linalg.norm(centered[:, 1])
        )
        correlation = (
            float(np.dot(centered[:, 0], centered[:, 1]) / denominator)
            if denominator > _EPSILON
            else 0.0
        )
        stereo = np.asarray(
            (
                channel_rms[0] / max(total, _EPSILON),
                channel_rms[1] / max(total, _EPSILON),
                correlation,
            )
        )
    tail = float(np.sqrt(np.mean(mono[len(mono) // 2 :] ** 2)))
    percussive = float((envelope[2] + envelope[3]) / max(tail, _EPSILON))
    return _Event(
        center=center,
        start=start,
        end=end,
        rms=rms,
        bands=bands,
        envelope=envelope,
        attack=attack_bins,
        decay=decay_bins,
        complex_attack=complex_attack,
        stereo=stereo,
        percussive=percussive,
    )


def _collect_events(audio: np.ndarray, sample_rate: int) -> list[_Event]:
    candidates: list[_Event] = []
    for _, block_start, block_end in _block_ranges(len(audio), sample_rate):
        block = np.asarray(audio[block_start:block_end], dtype=np.float64)
        centers = _event_centers(np.mean(block[:, :2], axis=1), sample_rate)
        for center in centers:
            event = _describe(audio, block_start + center, sample_rate)
            if event is not None:
                candidates.append(event)
    candidates.sort(key=lambda event: event.center)
    deduped: list[_Event] = []
    min_gap = max(1, round(sample_rate * _MIN_EVENT_SECONDS))
    for event in candidates:
        if deduped and event.center - deduped[-1].center < min_gap:
            if event.rms > deduped[-1].rms:
                deduped[-1] = event
            continue
        deduped.append(event)
        if len(deduped) >= _MAX_DESCRIPTORS:
            break
    return deduped


def _stereo_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left[2] < 0.0 or right[2] < 0.0:
        return 0.0
    return max(0.0, 1.0 - float(np.mean(np.abs(left - right))))


def _event_score(donor: _Event, exemplar: _Event) -> tuple[float, float] | None:
    spectral = _cosine(donor.bands, exemplar.bands)
    envelope = _cosine(donor.envelope, exemplar.envelope)
    attack = _cosine(donor.attack, exemplar.attack)
    decay = _cosine(donor.decay, exemplar.decay)
    stereo = _stereo_similarity(donor.stereo, exemplar.stereo)
    coherence = _signed_coherence(donor.complex_attack, exemplar.complex_attack)
    percussive = 1.0 - min(
        1.0, abs(np.log1p(donor.percussive) - np.log1p(exemplar.percussive))
    )
    if (
        spectral < _MIN_SPECTRAL
        or envelope < _MIN_ENVELOPE
        or attack < _MIN_ATTACK
        or decay < _MIN_DECAY
        or stereo < _MIN_STEREO
        or coherence < _MIN_COHERENCE
        or percussive < 0.6
    ):
        return None
    score = float(
        0.24 * spectral
        + 0.20 * envelope
        + 0.16 * attack
        + 0.14 * decay
        + 0.12 * stereo
        + 0.10 * coherence
        + 0.04 * percussive
    )
    return score, coherence


def _target_occupancy(target: np.ndarray, event: _Event) -> float:
    signal = np.asarray(target[event.start : event.end, :2], dtype=np.float64)
    return float(np.sqrt(np.mean(np.mean(signal, axis=1) ** 2))) if signal.size else 0.0


def _parent_support(
    parent: np.ndarray, donor: np.ndarray, event: _Event
) -> tuple[float, float]:
    parent_window = np.asarray(parent[event.start : event.end, :2], dtype=np.float64)
    donor_window = np.asarray(donor[event.start : event.end, :2], dtype=np.float64)
    parent_mono = np.mean(parent_window, axis=1)
    donor_mono = np.mean(donor_window, axis=1)
    parent_rms = (
        float(np.sqrt(np.mean(parent_mono * parent_mono))) if parent_mono.size else 0.0
    )
    donor_rms = (
        float(np.sqrt(np.mean(donor_mono * donor_mono))) if donor_mono.size else 0.0
    )
    if parent_rms <= _EPSILON or donor_rms <= _EPSILON:
        return 0.0, 0.0
    coherence = _signed_coherence(
        donor_mono.astype(np.complex128), parent_mono.astype(np.complex128)
    )
    return donor_rms / parent_rms, coherence


def _owner_ratio(
    events: Mapping[str, list[_Event]],
    donor: str,
    center: int,
    rms: float,
    sample_rate: int,
) -> float:
    nearby = [
        min(
            (
                event
                for event in events[name]
                if abs(event.center - center) <= round(0.03 * sample_rate)
            ),
            key=lambda event: abs(event.center - center),
            default=None,
        )
        for name in KIT_TARGETS
    ]
    total = sum(event.rms for event in nearby if event is not None)
    return rms / max(total, _EPSILON)


def _fade(length: int, sample_rate: int) -> np.ndarray:
    fade_length = min(length // 4, max(1, round(sample_rate * 0.008)))
    result = np.ones(length, dtype=np.float64)
    if fade_length:
        result[:fade_length] = np.linspace(0.0, 1.0, fade_length, endpoint=False)
        result[-fade_length:] = np.linspace(1.0, 0.0, fade_length, endpoint=False)
    return result


def _proposals(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
    events: Mapping[str, list[_Event]],
    sample_rate: int,
    arm: str,
) -> list[KitSiblingProposal]:
    proposals: list[KitSiblingProposal] = []
    min_gap = round(sample_rate * _MIN_EVENT_SECONDS)
    for donor in KIT_TARGETS:
        for donor_event in events[donor]:
            support, parent_coherence = _parent_support(
                parent, stems[donor], donor_event
            )
            if support < _MIN_PARENT_RATIO or parent_coherence <= 0.0:
                continue
            owner_ratio = _owner_ratio(
                events, donor, donor_event.center, donor_event.rms, sample_rate
            )
            if owner_ratio < _MIN_DONOR_RATIO:
                continue
            scores: dict[str, list[tuple[float, float, _Event]]] = {}
            for target in KIT_TARGETS:
                if (
                    target != donor
                    and _target_occupancy(stems[target], donor_event)
                    > donor_event.rms * _MAX_TARGET_RATIO
                ):
                    continue
                matches = []
                for exemplar in events[target]:
                    if abs(exemplar.center - donor_event.center) < min_gap:
                        continue
                    result = _event_score(donor_event, exemplar)
                    if result is not None:
                        score, coherence = result
                        matches.append((score, coherence, exemplar))
                matches.sort(key=lambda value: (-value[0], value[2].center))
                required = 2 if arm == "cue-plus-repetition" else 1
                if len(matches) >= required:
                    scores[target] = matches
            if not scores:
                continue
            ranked = sorted(
                ((values[0][0], target) for target, values in scores.items()),
                reverse=True,
            )
            best_score, target = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
            if target == donor or best_score - runner_up < _WINNER_MARGIN:
                continue
            matches = scores[target]
            exemplar_count = 1 if arm == "cue-only" else len(matches)
            selected = matches[:exemplar_count]
            score = float(np.mean([item[0] for item in selected]))
            coherence = float(np.mean([item[1] for item in selected]))
            proposals.append(
                KitSiblingProposal(
                    donor=donor,
                    target=target,
                    sample_start=donor_event.start,
                    sample_end=donor_event.end,
                    gain=_MAX_GAIN,
                    score=score,
                    signed_coherence=coherence,
                    exemplar_count=exemplar_count,
                    block_index=donor_event.center
                    // max(1, round(sample_rate * _BLOCK_SECONDS)),
                )
            )
    return proposals


def _accept_proposals(
    proposals: list[KitSiblingProposal],
) -> tuple[KitSiblingProposal, ...]:
    accepted: list[KitSiblingProposal] = []
    for proposal in sorted(
        proposals,
        key=lambda value: (-value.score, value.sample_start, value.donor, value.target),
    ):
        if len(accepted) >= _MAX_PROPOSALS:
            break
        if (
            sum(item.block_index == proposal.block_index for item in accepted)
            >= _MAX_PROPOSALS_PER_BLOCK
        ):
            continue
        overlaps = any(
            not (
                proposal.sample_end <= other.sample_start
                or proposal.sample_start >= other.sample_end
            )
            for other in accepted
        )
        if not overlaps:
            accepted.append(proposal)
    return tuple(
        sorted(
            accepted, key=lambda value: (value.sample_start, value.donor, value.target)
        )
    )


def repair_kit_siblings(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
    sample_rate: int = 44_100,
    arm: str = "cue-only",
) -> KitSiblingRepairResult:
    """Apply one bounded Q50 arm to six aligned DrumSep outputs.

    Descriptors are extracted from overlapping two-second blocks and discarded
    after a bounded event list is built. Proposals read only the original
    arrays. Accepted proposals move a faded slice of the donor's current
    waveform, subtracting it from the donor and adding it to the target.
    """
    normalized_arm = arm.replace("_", "-") if isinstance(arm, str) else arm
    checked = _validate_inputs(parent, stems, sample_rate, normalized_arm)
    output = {name: audio.copy() for name, audio in checked.items()}
    events = {name: _collect_events(checked[name], sample_rate) for name in KIT_TARGETS}
    counts = tuple((name, len(events[name])) for name in KIT_TARGETS)
    proposals = _accept_proposals(
        _proposals(parent, checked, events, sample_rate, normalized_arm)
    )
    for proposal in proposals:
        donor = np.asarray(
            checked[proposal.donor][proposal.sample_start : proposal.sample_end, :2],
            dtype=np.float64,
        )
        delta = donor * (_fade(len(donor), sample_rate)[:, None] * proposal.gain)
        donor_update = (
            np.asarray(
                output[proposal.donor][proposal.sample_start : proposal.sample_end],
                dtype=np.float64,
            )
            - delta
        )
        target_update = (
            np.asarray(
                output[proposal.target][proposal.sample_start : proposal.sample_end],
                dtype=np.float64,
            )
            + delta
        )
        if not np.all(np.isfinite(donor_update)) or not np.all(
            np.isfinite(target_update)
        ):
            continue
        output[proposal.donor][proposal.sample_start : proposal.sample_end, :2] = (
            donor_update.astype(output[proposal.donor].dtype)
        )
        output[proposal.target][proposal.sample_start : proposal.sample_end, :2] = (
            target_update.astype(output[proposal.target].dtype)
        )
    base_sum = sum(
        (np.asarray(audio, dtype=np.float64) for audio in checked.values()),
        np.zeros_like(parent, dtype=np.float64),
    )
    output_sum = sum(
        (np.asarray(audio, dtype=np.float64) for audio in output.values()),
        np.zeros_like(parent, dtype=np.float64),
    )
    conservation_error = float(np.max(np.abs(output_sum - base_sum)))
    if conservation_error > 1e-6 or not all(
        np.all(np.isfinite(audio)) for audio in output.values()
    ):
        output = {name: audio.copy() for name, audio in checked.items()}
        proposals = ()
        conservation_error = 0.0
    return KitSiblingRepairResult(
        output, proposals, normalized_arm, counts, conservation_error
    )


def evaluate_kit_sibling_arms(
    parent: np.ndarray,
    stems: Mapping[str, np.ndarray],
    sample_rate: int = 44_100,
) -> dict[str, KitSiblingRepairResult]:
    """Run both fixed arms so their transfer/repetition counts stay separate."""
    return {
        arm: repair_kit_siblings(parent, stems, sample_rate, arm) for arm in KIT_ARMS
    }


apply_kit_sibling_repair = repair_kit_siblings
