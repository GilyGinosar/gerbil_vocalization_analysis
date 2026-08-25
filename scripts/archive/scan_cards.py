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
from scripts.video.burrow_transit_picker import load_calls  # noqa: E402
from scripts.video.burrow_scan import AFTER_S, BEFORE_S, TILE_FPS, TILE_W  # noqa: E402
from scripts.video.burrow_transit_picker import (  # noqa: E402
    t_to_x,
    HTML_HEAD, HTML_TAIL, annotate_spectrogram, audio_path, file_index, mark_crossing,
    read_window, spectrogram_tile, time_axis, video_duration, write_html,
)

PX_PER_S = TILE_W * TILE_FPS      # the strips were cached at this scale


def shade(tile, t0, t1, a, b, reverse, delta):
    """Tint the span [a, b] of the card."""
    width = tile.shape[1]
    xa, xb = sorted((t_to_x(a, t0, t1, width, reverse), t_to_x(b, t0, t1, width, reverse)))
    xa, xb = max(0, xa), min(width, xb)
    if xb <= xa:
        return
    lifted = tile[:, xa:xb].astype(np.int16) + np.array(delta, np.int16)
    tile[:, xa:xb] = np.clip(lifted, 0, 255).astype(np.uint8)


GUIDE_KHZ = 15                    # movement noise below here, calls above


def presence_span(track: pd.DataFrame, entry: float, exit_: float) -> tuple[float, float]:
    """When the animal was in the tunnel at all, around this traverse.

    The 0.15/0.75 landmarks are where the measurement is trustworthy -- a body
    CENTROID cannot reach the crop edges, so wider landmarks detect almost
    nothing (0.05 is reached in 0.2% of frames). But the animal is in the tube
    for longer than the span between them, and that longer stretch is a real
    quantity: the contiguous run of single-animal frames containing the
    traverse. Shading it gives the eye the whole passage without pretending the
    landmarks are somewhere they are not.
    """
    single = track[(track.n_animals == 1) & track.x.notna()].frame.to_numpy()
    if not len(single):
        return entry, exit_
    a, b = int(entry * 30), int(exit_ * 30)
    inside = single[(single >= a) & (single <= b)]
    if not len(inside):
        return entry, exit_
    first, last = inside[0], inside[-1]
    while first - 1 in set(single[(single > first - 40) & (single < first)]):
        first -= 1
    prev = single[single < first]
    while len(prev) and first - prev[-1] == 1:
        first = prev[-1]; prev = prev[:-1]
    nxt = single[single > last]
    while len(nxt) and nxt[0] - last == 1:
        last = nxt[0]; nxt = nxt[1:]
    return first / 30.0, last / 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True, help="a burrow_scan output dir (holds traverses.csv + tiles/)")
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datadir")
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--include-multi", action="store_true")
    parser.add_argument("--alone", action="store_true",
                        help="keep only traverses where no second animal is in the tunnel at "
                             "any point in the card -- so a call cannot be aimed at a tube-mate")
    parser.add_argument("--left", type=float, default=0.15)
    parser.add_argument("--right", type=float, default=0.75)
    args = parser.parse_args()

    scan = Path(args.scan)
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")
    rows = [r for r in csv.DictReader(open(scan / "traverses.csv"))
            if r.get("tile") and (args.include_multi or r["single_animal"].lower() == "true")]

    calls_by_file = load_calls(args.exp)
    drift_cache: dict[int, float] = {}
    tracks: dict[str, pd.DataFrame] = {}
    cards = []
    dropped_not_alone = 0
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
        stem = Path(row["video"]).stem
        if stem not in tracks:
            tracks[stem] = pd.read_parquet(scan / "tracks" / f"{stem}.parquet")
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

        # No half-tile shift. Each tile shows the frame at the START of its 0.5 s
        # slot, and the lead-in is an exact multiple of that slot, so the entry
        # landmark lands precisely on a tile boundary -- the shading begins at the
        # left edge of the very frame in which the animal crosses. Centring the
        # frames would buy +/-0.25 s of nominal accuracy and lose that
        # correspondence, which is the thing you actually read off the card.
        if args.alone:
            # Strictly alone for the WHOLE card, not just under the 0.25 threshold and
            # not just during the traverse. If another animal is in the tube at all,
            # a call could be addressed to it rather than tied to the passage -- which
            # is the confound this filter exists to remove.
            window = tracks[stem]
            window = window[(window.frame >= t0 * 30) & (window.frame <= t1 * 30)]
            if len(window) and int(window.n_animals.max()) > 1:
                dropped_not_alone += 1
                continue

        audio, fs = read_window(audio_path(datadir, args.channel, index), t0, t1, drift, 0.0)
        spec = spectrogram_tile(audio, fs, width, reverse)
        # light tone = animal in the tunnel; strong tone + green lines = the traverse
        out = float(row["t_out"])
        # the two green lines are entry and the animal leaving the tunnel
        annotate_spectrogram(spec, t0, t1, entry, out, reverse)
        mark_crossing(spec, t0, t1, entry, out, BEFORE_S, ("enters", "out of tunnel"), reverse)
        axis = time_axis(width, t0, t1, entry, reverse)
        # a guide line where movement noise ends and calls begin
        y = int(spec.shape[0] * (1 - (GUIDE_KHZ * 1000 - 500) / (45000 - 500)))
        overlay = spec.copy()
        cv2.line(overlay, (0, y), (width, y), (200, 200, 200), 1)
        spec = cv2.addWeighted(overlay, 0.45, spec, 0.55, 0)
        cv2.putText(spec, f"{GUIDE_KHZ} kHz", (width - 60, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

        ok, buf = cv2.imencode(".jpg", cv2.vconcat([spec, strip, axis]),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        # DAS calls in view, so the sheets can be split into vocal and quiet
        counts: dict[str, int] = {}
        for call_start, _stop, event_type in calls_by_file.get(index, []):
            if t0 <= (call_start - 0.0) / drift <= t1:
                counts[event_type] = counts.get(event_type, 0) + 1
        call_text = ("  calls: " + ", ".join(f"{n} {t}" for t, n in sorted(counts.items()))
                     if counts else "  no calls")
        flag = "" if row["single_animal"].lower() == "true" else "  [>1 animal]"
        if row.get("still_in_tunnel_at_cap", "").lower() == "true":
            flag += "  [still in tunnel at +5s cap]"
        label = (f"{row['video']}  t={entry:.2f}s  {row['direction']}  "
                 f"traverse {float(row['traverse_s']):.1f}s  in tunnel "
                 f"{out - entry:.1f}s  window {t1 - t0:.1f}s{call_text}{flag}")
        cards.append((row["direction"], f"{row['video']}|{entry:.2f}",
                      "data:image/jpeg;base64," + base64.b64encode(buf).decode(), label))

    path = write_html(cards, Path(args.out_dir))
    if dropped_not_alone:
        print(f"dropped {dropped_not_alone} traverses with a second animal in the tunnel")
    print(f"{len(cards)} cards -> {path}  (no video decoded)")


if __name__ == "__main__":
    main()
