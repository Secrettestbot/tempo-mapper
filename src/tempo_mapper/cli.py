"""tempo-map command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .plot import save_plot
from .report import build_report
from .segments import change_points, detect_segments
from .tempo import analyze


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tempo-map",
        description="Detect BPM and tempo changes throughout a song.",
    )
    p.add_argument("song", help="audio file (mp3/ogg/flac/wav/...)")
    p.add_argument(
        "--mode", choices=["stable", "detailed", "both"], default="both",
        help="report style: smoothed segments, fine curve, or both (default)",
    )
    p.add_argument("--min-bpm", type=float, default=60.0,
                   help="low edge of the octave band (default 60)")
    p.add_argument("--max-bpm", type=float, default=200.0,
                   help="high edge of the octave band (default 200)")
    p.add_argument("--change-threshold", type=float, default=3.0,
                   help="BPM delta to count as a real tempo change (default 3)")
    p.add_argument("--min-segment", type=float, default=4.0,
                   help="minimum seconds for a segment to count (default 4)")
    p.add_argument("--smooth-seconds", type=float, default=2.0,
                   help="median-filter window for wobble removal (default 2)")
    p.add_argument("--round-to", type=float, default=1.0,
                   help="quantise reported BPM (e.g. 0.5, 0.1; default 1)")
    p.add_argument("--plot", nargs="?", const="", default=None, metavar="PATH",
                   help="save tempo plot PNG (default path if no value given)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the plot entirely")
    p.add_argument("--json", default=None, metavar="PATH",
                   help="also write machine-readable results (segments + curve)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isfile(args.song):
        print(f"error: file not found: {args.song}", file=sys.stderr)
        return 2

    try:
        analysis = analyze(
            args.song,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    segments = detect_segments(
        analysis.times,
        analysis.tempo_curve,
        change_threshold=args.change_threshold,
        min_segment=args.min_segment,
        smooth_seconds=args.smooth_seconds,
        round_to=args.round_to,
    )

    song_name = os.path.basename(args.song)
    print(build_report(analysis, segments, mode=args.mode, song_name=song_name))

    stem = os.path.splitext(args.song)[0]

    if not args.no_plot:
        plot_path = args.plot or f"{stem}_tempo.png"
        save_plot(analysis, segments, plot_path, song_name=song_name)
        print(f"Plot saved: {plot_path}")

    if args.json is not None:
        payload = {
            "song": song_name,
            "duration": analysis.duration,
            "dominant_bpm": analysis.dominant_bpm,
            "offset": analysis.offset,
            "min_bpm": analysis.min_bpm,
            "max_bpm": analysis.max_bpm,
            "segments": [
                {"start": s.start, "end": s.end, "bpm": s.bpm,
                 "duration": s.duration}
                for s in segments
            ],
            "change_points": [
                {"time": t, "bpm": b} for t, b in change_points(segments)
            ],
            "fine_curve": {
                "times": [round(float(t), 4) for t in analysis.times],
                "bpm": [round(float(v), 4) for v in analysis.tempo_curve],
            },
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON saved: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
