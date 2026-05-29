"""Audio loading. librosa uses soundfile/audioread (ffmpeg) under the hood,
so mp3/ogg/flac/wav all decode transparently."""

from __future__ import annotations

import librosa
import numpy as np


def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load ``path`` as mono at ``sr`` Hz. Returns (samples, sr)."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"No audio samples decoded from {path!r}")
    return y, sr
