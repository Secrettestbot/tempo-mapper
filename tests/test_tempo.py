"""End-to-end tests on synthetic click tracks (validates the real pipeline)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf

from synth import click_track
from tempo_mapper.segments import detect_segments
from tempo_mapper.tempo import analyze


def _write(samples, sr=22050):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, samples, sr)
    return path


def test_fixed_bpm_dominant():
    path = _write(click_track([(128.0, 30.0)]))
    try:
        a = analyze(path, min_bpm=60, max_bpm=200)
        assert abs(a.dominant_bpm - 128.0) <= 2.0
    finally:
        os.remove(path)


def test_tempo_switch_detected():
    # 20 s at 120, then 20 s at 150 — the headline feature.
    path = _write(click_track([(120.0, 20.0), (150.0, 20.0)]))
    try:
        a = analyze(path, min_bpm=60, max_bpm=200)
        segs = detect_segments(
            a.times, a.tempo_curve, change_threshold=4.0, min_segment=5.0
        )
        bpms = [s.bpm for s in segs]
        assert len(segs) >= 2, f"expected >=2 segments, got {bpms}"
        # A ~120 segment early and a ~150 segment later.
        assert any(abs(b - 120) <= 5 for b in bpms), bpms
        assert any(abs(b - 150) <= 5 for b in bpms), bpms
        # A change point somewhere around the 20 s boundary (allow lag).
        boundaries = [s.start for s in segs[1:]]
        assert any(15.0 < t < 25.0 for t in boundaries), boundaries
    finally:
        os.remove(path)
