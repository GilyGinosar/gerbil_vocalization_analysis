#!/usr/bin/env python
"""Build curation cards from a burrow_scan run -- spectrogram over the cached frame strip.

`burrow_scan.py` caches a frame strip per traverse while the video is already
decoded. This turns those strips into cards without touching video again: read
the strip, compute the tunnel-mic spectrogram for exactly the same window, stack
them on one time axis. Seconds instead of a ten-minute re-decode, which is the
whole point of caching.

Time runs the way the animal does -- left-to-right for `to_arena`, right-to-left
for `to_nest` -- with t=0 at the moment it enters the tunnel.

    python scripts/video/scan_cards.py --scan exports/scan_2026_02/492 --exp 492 \
        --out-dir exports/scan_2026_02/492/cards
"""
from __future__ import annotations

import argparse
import base64
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pipeline.paths import BASE_RAW  # noqa: E402
from scripts.video.burrow_scan import AFTER_S, BEFORE_S, TILE_FPS, TILE_W  # noqa: E402
from scripts.video.burrow_transit_picker import (  # noqa: E402
    HTML_HEAD, HTML_TAIL, annotate_spectrogram, audio_path, file_index, mark_crossing,
    read_window, spectrogram_tile, time_axis, video_duration, write_html,
)

PX_PER_S = TILE_W * TILE_FPS      # the strips were cached at this scale
POS_H = 70                        # px for the position trace
GUIDE_KHZ = 15                    # movement noise below here, calls above


def position_trace(track: pd.DataFrame, t0: float, t1: float, width: int,
                   left: float, right: float, reverse: bool) -> np.ndarray:
    """The animal's x through the tunnel, at full 30 fps, on the card's time axis.

    The frame strip samples at 2 fps because each tile costs half a second of
    width. The track has every frame and costs nothing, so this shows the actual
    trajectory -- and the landmark crossings become visible as the curve cutting
    the two guide lines, instead of something you take on trust.
    """
    from scripts.video.burrow_transit_picker import t_to_x
    strip = np.full((POS_H, width, 3), 18, np.uint8)
    for level, label in ((left, "0.15 nest"), (right, "0.75 arena")):
        y = int(POS_H * (1 - level))
        for x in range(0, width, 12):
            cv2.line(strip, (x, y), (min(x + 6, width), y), (70, 110, 70), 1)
        cv2.putText(strip, label, (4 if not reverse else width - 78, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 140, 90), 1)
    window = track[(track.frame >= t0 * 30) & (track.frame <= t1 * 30)]
    previous = None
    for frame, n_animals, x in zip(window.frame, window.n_animals, window.x):
        if n_animals != 1 or not np.isfinite(x):
            previous = None
            continue
        px = t_to_x(frame / 30.0, t0, t1, width, reverse)
        py = int(POS_H * (1 - float(x)))
        if previous is not None:
            cv2.line(strip, previous, (px, py), (255, 220, 120), 2)
        previous = (px, py)
    return strip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True, help="a burrow_scan output dir (holds traverses.csv + tiles/)")
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datadir")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--include-multi", action="store_true")
    parser.add_argument("--left", type=float, default=0.15)
    parser.add_argument("--right", type=float, default=0.75)
    args = parser.parse_args()

    scan = Path(args.scan)
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")
    rows = [r for r in csv.DictReader(open(scan / "traverses.csv"))
            if r.get("tile") and (args.include_multi or r["single_animal"].lower() == "true")]

    drift_cache: dict[int, float] = {}
    cards = []
    for row in rows:
        strip = cv2.imread(str(scan / row["tile"]))
        if strip is None:
            continue
        index = file_index(row["video"])
        if index not in drift_cache:
            wav = audio_path(datadir, args.channel, index)
            cap = cv2.VideoCapture(str(datadir / row["video"]))
            with sf.SoundFile(str(wav)) as handle:
                audio_dur = handle.frames / handle.samplerate
            drift_cache[index] = audio_dur / video_duration(cap)
            cap.release()
        drift = drift_cache[index]

        entry, exit_ = float(row["t_entry"]), float(row["t_exit"])
        width = strip.shape[1]
        t0 = entry - BEFORE_S
        t1 = t0 + width / PX_PER_S          # the strip's own extent defines the window
        reverse = row["direction"] == "to_nest"

        if reverse:
            # The cached strip is always in forward time order. The spectrogram and
            # ruler are mirrored for to_nest, so the tiles must be re-ordered to
            # match -- re-ordered, not mirrored: each frame keeps its own geometry
            # so the nest stays on the left inside every tile.
            n_tiles = width // TILE_W
            tiles = [strip[:, i * TILE_W:(i + 1) * TILE_W] for i in range(n_tiles)]
            strip = cv2.hconcat(tiles[::-1])

        # Each cached tile shows the frame at the START of its 0.5 s slot, so the
        # frame beside a landmark could be up to half a second stale. Shifting the
        # strip left by half a tile puts each frame's instant at its tile's centre,
        # making the error symmetric (+/-0.25 s) instead of one-sided.
        shift = TILE_W // 2
        strip = cv2.hconcat([strip[:, shift:], np.zeros((strip.shape[0], shift, 3), np.uint8)])

        audio, fs = read_window(audio_path(datadir, args.channel, index), t0, t1, drift, 0.0)
        spec = spectrogram_tile(audio, fs, width, reverse)
        annotate_spectrogram(spec, t0, t1, entry, exit_, reverse)
        mark_crossing(spec, t0, t1, entry, exit_, BEFORE_S, ("enters", "arrives"), reverse)
        axis = time_axis(width, t0, t1, entry, reverse)
        track = pd.read_parquet(scan / "tracks" / f"{Path(row['video']).stem}.parquet")
        trace = position_trace(track, t0, t1, width, args.left, args.right, reverse)
        # a guide line where movement noise ends and calls begin
        y = int(spec.shape[0] * (1 - (GUIDE_KHZ * 1000 - 500) / (45000 - 500)))
        overlay = spec.copy()
        cv2.line(overlay, (0, y), (width, y), (200, 200, 200), 1)
        spec = cv2.addWeighted(overlay, 0.45, spec, 0.55, 0)
        cv2.putText(spec, f"{GUIDE_KHZ} kHz", (width - 60, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        ok, buf = cv2.imencode(".jpg", cv2.vconcat([spec, trace, strip, axis]),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        flag = "" if row["single_animal"].lower() == "true" else "  [>1 animal]"
        label = (f"{row['video']}  t={entry:.2f}s  {row['direction']}  "
                 f"traverse {float(row['traverse_s']):.1f}s  window {t1 - t0:.1f}s{flag}")
        cards.append((row["direction"], f"{row['video']}|{entry:.2f}",
                      "data:image/jpeg;base64," + base64.b64encode(buf).decode(), label))

    path = write_html(cards, Path(args.out_dir))
    print(f"{len(cards)} cards -> {path}  (no video decoded)")


if __name__ == "__main__":
    main()
