"""Turn a noisy per-frame tempo curve into stable segments and the
(timestamp, BPM) change points a rhythm game wants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt


@dataclass
class Segment:
    start: float   # seconds
    end: float     # seconds
    bpm: float     # representative BPM (median of the segment, rounded)
    duration: float


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _smooth(curve: np.ndarray, dt: float, smooth_seconds: float) -> np.ndarray:
    """Median filter to kill momentary wobble without shifting real changes."""
    if smooth_seconds <= 0 or dt <= 0:
        return np.asarray(curve, dtype=float)
    kernel = _odd(round(smooth_seconds / dt))
    if kernel <= 1 or kernel >= curve.size:
        return np.asarray(curve, dtype=float)
    return medfilt(np.asarray(curve, dtype=float), kernel_size=kernel)


def detect_segments(
    times: np.ndarray,
    tempo_curve: np.ndarray,
    change_threshold: float = 3.0,
    min_segment: float = 4.0,
    smooth_seconds: float = 2.0,
    round_to: float = 1.0,
) -> list[Segment]:
    """Segment ``tempo_curve`` into runs of roughly-constant tempo.

    A new segment opens when the smoothed tempo diverges by more than
    ``change_threshold`` BPM from the current segment's median. Segments shorter
    than ``min_segment`` seconds are absorbed into their nearest-BPM neighbour,
    then adjacent near-equal segments are merged. ``round_to`` quantises the
    reported BPM (use 0.5 / 0.1 for finer game timing points)."""
    times = np.asarray(times, dtype=float)
    curve = np.asarray(tempo_curve, dtype=float)
    n = curve.size
    if n == 0:
        return []
    dt = float(np.median(np.diff(times))) if n > 1 else 0.0
    smoothed = _smooth(curve, dt, smooth_seconds)

    # Pass 1: grow raw segments off the smoothed curve.
    groups: list[list[int]] = [[0]]
    refs: list[float] = [smoothed[0]]
    for i in range(1, n):
        if abs(smoothed[i] - refs[-1]) <= change_threshold:
            groups[-1].append(i)
            refs[-1] = float(np.median(smoothed[groups[-1]]))
        else:
            groups.append([i])
            refs.append(smoothed[i])

    segs = [_make_segment(g, times, smoothed, dt, n, round_to) for g in groups]

    # Pass 2: absorb too-short segments into the closest-BPM neighbour.
    segs = _absorb_short(segs, min_segment, times, smoothed, dt, n, groups, round_to)

    # Pass 3: merge adjacent segments whose BPMs are within threshold.
    segs = _merge_adjacent(segs, change_threshold)
    return segs


def _make_segment(group, times, smoothed, dt, n, round_to) -> Segment:
    start = float(times[group[0]])
    last = group[-1]
    end = float(times[last] + dt) if last < n - 1 else float(times[last] + dt)
    bpm = round(float(np.median(smoothed[group])) / round_to) * round_to
    return Segment(start=start, end=end, bpm=bpm, duration=end - start)


def _absorb_short(segs, min_segment, times, smoothed, dt, n, groups, round_to):
    """Iteratively merge the shortest sub-threshold segment into a neighbour.

    Works on the index groups so the merged segment's BPM is recomputed from all
    its frames (not an average of segment summaries)."""
    groups = [list(g) for g in groups]
    while len(groups) > 1:
        durs = [
            (times[g[-1]] + dt) - times[g[0]] for g in groups
        ]
        order = sorted(range(len(groups)), key=lambda k: durs[k])
        shortest = next((k for k in order if durs[k] < min_segment), None)
        if shortest is None:
            break
        bpm_k = np.median(smoothed[groups[shortest]])
        # Pick the adjacent group with the closest BPM.
        neighbours = []
        if shortest > 0:
            neighbours.append(shortest - 1)
        if shortest < len(groups) - 1:
            neighbours.append(shortest + 1)
        best = min(
            neighbours, key=lambda k: abs(np.median(smoothed[groups[k]]) - bpm_k)
        )
        lo, hi = sorted((shortest, best))
        merged = groups[lo] + groups[hi]
        groups[lo:hi + 1] = [merged]
    return [_make_segment(g, times, smoothed, dt, n, round_to) for g in groups]


def _merge_adjacent(segs: list[Segment], change_threshold: float) -> list[Segment]:
    if not segs:
        return segs
    out = [segs[0]]
    for s in segs[1:]:
        prev = out[-1]
        if abs(s.bpm - prev.bpm) <= change_threshold:
            # Weighted-average BPM, extend the span.
            total = prev.duration + s.duration
            bpm = (prev.bpm * prev.duration + s.bpm * s.duration) / total
            out[-1] = Segment(
                start=prev.start, end=s.end,
                bpm=round(bpm, 2), duration=s.end - prev.start,
            )
        else:
            out.append(s)
    return out


def change_points(segments: list[Segment]) -> list[tuple[float, float]]:
    """(timestamp, BPM) at each segment boundary — the rhythm-game timing points."""
    return [(round(s.start, 3), s.bpm) for s in segments]
