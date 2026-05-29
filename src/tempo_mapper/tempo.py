"""Core tempo analysis: onset envelope -> per-frame tempo curve, dominant BPM,
first-beat offset. Built on librosa 0.11."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .audio import load_audio


@dataclass
class TempoAnalysis:
    times: np.ndarray        # seconds, one per frame
    tempo_curve: np.ndarray  # per-frame BPM, octave-folded into [min_bpm, max_bpm]
    dominant_bpm: float      # duration-weighted modal BPM of the curve
    offset: float            # time (s) of the first detected beat
    duration: float          # total audio length (s)
    sr: int
    hop_length: int
    min_bpm: float
    max_bpm: float


def fold_to_band(bpm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fold tempo octaves (x2 / /2) so values land inside [lo, hi].

    Resolves the classic half/double-time ambiguity (e.g. 70 vs 140) by mapping
    every estimate into one consistent band. Works element-wise on arrays.
    Assumes the band spans at least one octave (hi >= 2*lo)."""
    out = np.asarray(bpm, dtype=float).copy()
    pos = out > 0
    # Bring up values below the band, push down values above it.
    while np.any(pos & (out < lo)):
        m = pos & (out < lo)
        out[m] *= 2
    while np.any(pos & (out > hi)):
        m = pos & (out > hi)
        out[m] /= 2
    return out


def dominant_bpm(tempo_curve: np.ndarray, bin_width: float = 1.0) -> float:
    """Duration-weighted modal BPM. Every frame is equal duration, so this is
    just the most common rounded value across the curve."""
    valid = tempo_curve[np.isfinite(tempo_curve) & (tempo_curve > 0)]
    if valid.size == 0:
        return 0.0
    binned = np.round(valid / bin_width) * bin_width
    values, counts = np.unique(binned, return_counts=True)
    return float(values[np.argmax(counts)])


def analyze(
    path: str,
    sr: int = 22050,
    hop_length: int = 512,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
) -> TempoAnalysis:
    """Decode ``path`` and estimate a per-frame tempo curve plus a dominant BPM."""
    if max_bpm < 2 * min_bpm:
        raise ValueError(
            f"--max-bpm ({max_bpm}) must be at least 2x --min-bpm ({min_bpm}) "
            "so octave folding has an unambiguous target band."
        )

    y, sr = load_audio(path, sr=sr)
    duration = float(len(y) / sr)

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Per-frame ("dynamic") tempo estimate — the fine-detail curve.
    # start_bpm centres the prior in the middle of the requested band.
    start_bpm = float(np.sqrt(min_bpm * max_bpm))
    dtempo = librosa.feature.tempo(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        aggregate=None,
        start_bpm=start_bpm,
        max_tempo=max_bpm * 2,  # let it see double-time, then fold below
    )
    tempo_curve = np.squeeze(np.asarray(dtempo, dtype=float))
    if tempo_curve.ndim > 1:
        tempo_curve = tempo_curve[0]
    tempo_curve = fold_to_band(tempo_curve, min_bpm, max_bpm)

    times = librosa.times_like(tempo_curve, sr=sr, hop_length=hop_length)

    # First-beat offset (rhythm games usually need this alongside the BPM).
    _, beats = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=hop_length, start_bpm=start_bpm
    )
    offset = (
        float(librosa.frames_to_time(beats[0], sr=sr, hop_length=hop_length))
        if len(beats)
        else 0.0
    )

    return TempoAnalysis(
        times=times,
        tempo_curve=tempo_curve,
        dominant_bpm=dominant_bpm(tempo_curve),
        offset=offset,
        duration=duration,
        sr=sr,
        hop_length=hop_length,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
