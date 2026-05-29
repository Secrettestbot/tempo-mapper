"""Synthetic click-track generation for tests."""

from __future__ import annotations

import numpy as np


def click(sr: int, freq: float = 1000.0, dur: float = 0.01) -> np.ndarray:
    """A short percussive click: a decaying sine burst (fires onset detection)."""
    t = np.arange(int(sr * dur)) / sr
    env = np.exp(-t * 400.0)
    return np.sin(2 * np.pi * freq * t) * env


def click_track(bpm_sections, sr: int = 22050) -> np.ndarray:
    """Build a click track from ``[(bpm, seconds), ...]`` sections.

    Beats are spaced at 60/bpm; each beat is a short click. Returns mono float32."""
    one = click(sr)
    total = int(sum(sec for _, sec in bpm_sections) * sr)
    out = np.zeros(total + one.size, dtype=np.float32)
    pos = 0  # running sample offset at each section boundary
    for bpm, sec in bpm_sections:
        interval = sr * 60.0 / bpm
        n_beats = int(sec * sr / interval)
        for b in range(n_beats):
            i = pos + int(round(b * interval))
            out[i:i + one.size] += one
        pos += int(sec * sr)
    return out[:total]
