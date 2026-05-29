# Tempo Mapper

Automatic BPM and **tempo-change** detection for rhythm-game songs. Point it at an
audio file and it tells you the dominant BPM, where the tempo changes, and gives
you a ready-to-paste list of timing points — plus a plot of tempo over time.

## Install

```bash
cd ~/Documents/Tempo_Mapper
python3 -m venv .venv
.venv/bin/pip install -e .
```

Requires `ffmpeg` on the system (used to decode mp3/ogg/flac/etc.).

## Usage

```bash
.venv/bin/tempo-map song.mp3
```

Example output:

```
Dominant BPM   : 128
First beat     : 0.627 s (offset)
Tempo changes  : 2 detected

Stable segments
   start       end    length      BPM
 0:00.00   0:17.86   0:17.86      100
 0:17.86   0:35.69   0:17.83      128
 0:35.69   0:54.01   0:18.32      150

Timing points (paste into the game)
     0.000s  ->  100 BPM
    17.856s  ->  128 BPM
    35.689s  ->  150 BPM
```

A tempo plot is saved next to the song (`song_tempo.png`).

### Options

| Flag | Meaning | Default |
|------|---------|---------|
| `--mode stable\|detailed\|both` | smoothed segments, fine curve, or both | `both` |
| `--min-bpm` / `--max-bpm` | octave band; folds half/double-time into one range | `60` / `200` |
| `--change-threshold` | BPM delta to count as a real change | `3` |
| `--min-segment` | minimum seconds for a segment to count | `4` |
| `--smooth-seconds` | median-filter window to ignore wobble | `2` |
| `--round-to` | quantise reported BPM (`0.5`, `0.1`, …) | `1` |
| `--plot [PATH]` / `--no-plot` | where to save / skip the plot | next to song |
| `--json PATH` | also write machine-readable segments + fine curve | off |

## Tips

- **Wrong octave?** If it reports half or double the real tempo, narrow the band
  (e.g. `--min-bpm 100 --max-bpm 200`).
- **Too many / too few changes?** Raise `--change-threshold` and `--min-segment`
  to ignore minor drift; lower them to catch subtle shifts.
- Change timestamps can lag the true change by ~1–2 s (the smoothing window);
  nudge them in-game if needed.

## How it works

librosa onset envelope → per-frame ("dynamic") tempo curve → octave-folded into
your band → median-smoothed → segmented into runs of roughly-constant tempo. The
segment boundaries are your timing points. See `src/tempo_mapper/`.

## Tests

```bash
.venv/bin/pytest
```

Synthetic click tracks validate both fixed-BPM detection and a known mid-song
tempo switch.
