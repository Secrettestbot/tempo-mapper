"""Human-readable report: dominant BPM, offset, stable-segment table, and a
ready-to-transcribe timing-points block."""

from __future__ import annotations

import numpy as np

from .segments import Segment, change_points
from .tempo import TempoAnalysis


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60)
    return f"{int(m):d}:{s:05.2f}"


def build_report(
    analysis: TempoAnalysis,
    segments: list[Segment],
    mode: str = "both",
    song_name: str = "",
) -> str:
    lines: list[str] = []
    title = f"Tempo report — {song_name}" if song_name else "Tempo report"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Duration       : {fmt_time(analysis.duration)}")
    lines.append(f"Dominant BPM   : {analysis.dominant_bpm:g}")
    lines.append(f"First beat     : {analysis.offset:.3f} s (offset)")
    lines.append(f"Analysis band  : {analysis.min_bpm:g}–{analysis.max_bpm:g} BPM")
    changes = len(segments) - 1 if segments else 0
    lines.append(f"Tempo changes  : {changes} detected")
    lines.append("")

    if mode in ("stable", "both"):
        lines.append("Stable segments")
        lines.append("-" * 15)
        lines.append(f"{'start':>8}  {'end':>8}  {'length':>8}  {'BPM':>7}")
        for s in segments:
            lines.append(
                f"{fmt_time(s.start):>8}  {fmt_time(s.end):>8}  "
                f"{fmt_time(s.duration):>8}  {s.bpm:>7g}"
            )
        lines.append("")
        lines.append("Timing points (paste into the game)")
        lines.append("-" * 35)
        for t, bpm in change_points(segments):
            lines.append(f"  {t:8.3f}s  ->  {bpm:g} BPM")
        lines.append("")

    if mode in ("detailed", "both"):
        curve = analysis.tempo_curve
        valid = curve[np.isfinite(curve) & (curve > 0)]
        lines.append("Fine tempo curve (moment-to-moment)")
        lines.append("-" * 35)
        if valid.size:
            lines.append(
                f"  min {valid.min():g} | median {np.median(valid):g} | "
                f"max {valid.max():g} | std {valid.std():.1f} BPM"
            )
        lines.append("  Full per-frame curve is in the plot (--plot) and JSON (--json).")
        lines.append("")

    return "\n".join(lines)
