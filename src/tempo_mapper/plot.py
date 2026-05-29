"""Save a tempo-over-time PNG: fine curve + stable-segment steps + dominant line."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

from .segments import Segment
from .tempo import TempoAnalysis


def save_plot(
    analysis: TempoAnalysis,
    segments: list[Segment],
    out_path: str,
    song_name: str = "",
) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        analysis.times, analysis.tempo_curve,
        color="#b0b0b0", lw=0.8, alpha=0.9,
        label="fine tempo curve",
    )

    # Stable segments as a bold step function.
    for i, s in enumerate(segments):
        ax.hlines(
            s.bpm, s.start, s.end,
            color="#1f77b4", lw=3,
            label="stable segments" if i == 0 else None,
        )
        if i > 0:  # vertical marker at each change point
            ax.axvline(s.start, color="#1f77b4", lw=0.8, ls=":", alpha=0.6)

    ax.axhline(
        analysis.dominant_bpm, color="#d62728", lw=1, ls="--",
        label=f"dominant {analysis.dominant_bpm:g} BPM",
    )

    ax.set_xlim(0, analysis.duration)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("BPM")
    ax.set_title(f"Tempo map — {song_name}" if song_name else "Tempo map")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
