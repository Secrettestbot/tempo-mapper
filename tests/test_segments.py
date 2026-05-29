"""Unit tests for the segmentation logic (no audio, fast)."""

from __future__ import annotations

import numpy as np

from tempo_mapper.segments import change_points, detect_segments


def _times(n, dt=0.1):
    return np.arange(n) * dt


def test_constant_curve_is_one_segment():
    curve = np.full(300, 128.0)
    segs = detect_segments(_times(300), curve, min_segment=2.0)
    assert len(segs) == 1
    assert abs(segs[0].bpm - 128.0) < 1e-6


def test_single_change_detected():
    # 15 s at 120, then 15 s at 150 (dt=0.1 -> 150 frames each).
    curve = np.concatenate([np.full(150, 120.0), np.full(150, 150.0)])
    segs = detect_segments(_times(300), curve, change_threshold=3.0, min_segment=4.0)
    assert len(segs) == 2
    assert abs(segs[0].bpm - 120.0) < 2
    assert abs(segs[1].bpm - 150.0) < 2
    # Change point near the 15 s boundary.
    cps = change_points(segs)
    assert len(cps) == 2
    assert 14.0 < cps[1][0] < 16.0


def test_noise_spikes_are_absorbed():
    # Constant 100 BPM with isolated single-frame spikes — must stay one segment.
    curve = np.full(300, 100.0)
    curve[50] = 160.0
    curve[123] = 40.0
    curve[200] = 175.0
    segs = detect_segments(
        _times(300), curve, change_threshold=3.0, min_segment=4.0, smooth_seconds=1.0
    )
    assert len(segs) == 1
    assert abs(segs[0].bpm - 100.0) < 2


def test_short_segment_absorbed_into_neighbour():
    # A 1 s blip at 150 inside a long 120 stretch should be absorbed (min_segment=4).
    curve = np.concatenate(
        [np.full(100, 120.0), np.full(10, 150.0), np.full(100, 120.0)]
    )
    segs = detect_segments(
        _times(210), curve, change_threshold=3.0, min_segment=4.0, smooth_seconds=0.5
    )
    assert len(segs) == 1
    assert abs(segs[0].bpm - 120.0) < 2


def test_empty_curve():
    assert detect_segments(np.array([]), np.array([])) == []
