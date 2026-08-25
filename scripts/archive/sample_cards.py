#!/usr/bin/env python
"""Sample traverses from across a whole date folder and build cards from cached tiles.

`scan_cards.py` works on one experiment. After a date-folder scan there are
thousands of traverses across 60 experiments, and eyeballing needs a
*representative* handful rather than one experiment's worth -- so this samples
evenly across experiments and splits the sample by whether the tunnel mic heard
anything.

"With calls" means a tunnel-localised call in the card window, using each
experiment's own threshold -- not merely any call in earshot, which mostly
tracks whether the colony happened to be noisy.

    python scripts/video/sample_cards.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --per-cell 15 --out-dir exports/sample_2026_02
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
from scripts.pipeline.paths import experiment_audio_dir  # noqa: E402
from scripts.video.burrow_transit_picker import CALL_COLORS  # noqa: E402
from scripts.video.burrow_transit_picker import (  # noqa: E402
    annotate_spectrogram, audio_path, mark_crossing, read_window,
    spectrogram_tile, time_axis, video_duration, write_html,
)

PX_PER_S = TILE_W * TILE_FPS
GUIDE_KHZ = 15
RIBBON_H = 13


def all_calls(exp: int) -> dict[str, dict[int, list]]:
    """Every DAS call, per compartment, with its type -- not just the USV subset.

    Drawn under the spectrogram so the detector can be checked against the audio:
    a sweep with no tick is a miss, a tick with no sweep is a false positive, and
    a tick in the wrong compartment is a location-assignment error. All three
    would propagate silently into every rate we have computed.
    """
    path = experiment_audio_dir(exp) / "calls.csv"
    out: dict[str, dict[int, list]] = {"underground": {}, "arena_1": {}}
    if not path.exists():
        return out
    with open(path) as handle:
        for row in csv.DictReader(handle):
            loc = row["assigned_location"]
            if loc in out:
                out[loc].setdefault(int(row["file_num"]), []).append(
                    (float(row["start_time_file_sec"]),
                     float(row["stop_time_file_sec"]), row["event_type"]))
    return out


def ribbon(calls: list, t0: float, t1: float, width: int, reverse: bool,
           label: str) -> np.ndarray:
    """One row of DAS detections, coloured by call type."""
    from scripts.video.burrow_transit_picker import t_to_x
    bar = np.full((RIBBON_H, width, 3), 24, np.uint8)
    n = 0
    for start, stop, event_type in calls:
        if stop < t0 or start > t1:
            continue
        n += 1
        xa, xb = sorted((t_to_x(start, t0, t1, width, reverse),
                         t_to_x(stop, t0, t1, width, reverse)))
        xa, xb = max(0, xa), min(width, max(xb, xa + 2))
        cv2.rectangle(bar, (xa, 2), (xb, RIBBON_H - 3),
                      CALL_COLORS.get(event_type, (180, 180, 180)), -1)
    cv2.putText(bar, f"{label} ({n})", (4, RIBBON_H - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1)
    return bar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--per-cell", type=int, default=15,
                        help="traverses per direction x (calls / quiet) cell")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--direction", help="keep only this direction")
    parser.add_argument("--position-band", help="lo,hi -- select traverses whose calls fall in "
                                                "this stretch of tunnel, e.g. 0.05,0.15")
    parser.add_argument("--min-in-band", type=int, default=2,
                        help="how many calls inside the band a traverse needs to qualify")
    args = parser.parse_args()

    scan = Path(args.scan)
    traverses = pd.read_parquet(scan / f"traverses_{args.date}.parquet")
    traverses = traverses[traverses.single_animal & traverses.tile.notna()]

    # per-experiment thresholds and the tunnel-localised calls
    localised = {}
    for path in sorted((scan / "origin").glob("*/call_origin.csv")):
        exp = int(path.parent.name)
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        reference = table.loc[table.state == "tunnel empty", "tunnel_db_over_nest"]
        if len(reference) < 50:
            continue
        hit = table[table.tunnel_db_over_nest > reference.quantile(0.95)]
        for file_num, group in hit.groupby("file"):
            localised[(exp, int(file_num))] = np.sort(group.start_s.to_numpy())

    def n_localised(row) -> int:
        times = localised.get((row.exp, row.file_num))
        if times is None:
            return 0
        t0, t1 = row.t_entry - BEFORE_S, row.t_out + AFTER_S
        return int(((times >= t0) & (times <= t1)).sum())

    traverses = traverses[[(e, f) in localised for e, f in zip(traverses.exp, traverses.file_num)]]
    traverses["n_tunnel_calls"] = [n_localised(r) for r in traverses.itertuples()]
    print(f"{len(traverses)} traverses in experiments with a usable threshold; "
          f"{int((traverses.n_tunnel_calls > 0).sum())} have a tunnel-localised call")

    if args.position_band:
        lo, hi = (float(v) for v in args.position_band.split(","))
        if args.direction:
            traverses = traverses[traverses.direction == args.direction]
        # count each traverse's calls that occur while the animal is inside the band
        from scripts.video.burrow_transit_picker import load_calls as _load
        FPS = 30
        in_band = []
        cache: dict[int, dict] = {}
        for row in traverses.itertuples():
            if row.exp not in cache:
                calls = _load(int(row.exp))
                cache[row.exp] = {k: np.sort(np.array([c for c, _, t in v
                                                       if t in ("high-freq", "warble")]))
                                  for k, v in calls.items()}
            times = cache[row.exp].get(row.file_num)
            n = 0
            if times is not None and times.size:
                tp = scan / str(row.exp) / "tracks" / f"{Path(row.video).stem}.parquet"
                if tp.exists():
                    track = pd.read_parquet(tp)
                    xs, na = track.x.to_numpy(), track.n_animals.to_numpy()
                    a, b = int(row.t_entry * FPS), int(row.t_out * FPS)
                    sel = times[(times >= row.t_entry) & (times <= row.t_out)]
                    if sel.size and b < len(xs):
                        idx = np.clip((sel * FPS).astype(int), a, max(a, b - 1))
                        px = xs[idx]
                        n = int(((px >= lo) & (px <= hi) & np.isfinite(px)
                                 & (na[idx] == 1)).sum())
            in_band.append(n)
        traverses = traverses.assign(n_in_band=in_band)
        qualified = traverses[traverses.n_in_band >= args.min_in_band]
        print(f"{len(qualified)} traverses with >= {args.min_in_band} calls in "
              f"position {lo}-{hi} (of {len(traverses)})")
        order = qualified.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
        chosen = qualified.assign(rank=order).sort_values(
            ["rank", "n_in_band"], ascending=[True, False]).head(args.per_cell)
        picked = [(r.direction, f"band{lo}-{hi}", r) for r in chosen.itertuples()]
        print(f"  drawing {len(picked)} from {chosen.exp.nunique()} experiments")
    else:
        picked = []
        for direction in ("to_arena", "to_nest"):
         for label, mask in (("calls", traverses.n_tunnel_calls > 0),
                            ("quiet", traverses.n_tunnel_calls == 0)):
            pool = traverses[(traverses.direction == direction) & mask]
            if pool.empty:
                continue
            # spread across experiments: one per experiment, cycling, rather than
            # 15 from whichever experiment happens to be biggest
            order = pool.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
            chosen = pool.assign(rank=order).sort_values(["rank", "exp"]).head(args.per_cell)
            for row in chosen.itertuples():
                picked.append((direction, label, row))
            print(f"  {direction:<9} {label:<6} {len(chosen):3d} from "
                  f"{chosen.exp.nunique()} experiments")

    das_cache: dict[int, dict] = {}
    drift_cache: dict[tuple[int, int], float] = {}
    cards = []
    for direction, label, row in picked:
        strip = cv2.imread(str(scan / str(row.exp) / row.tile))
        if strip is None:
            continue
        datadir = BASE_RAW / f"experiment_{row.exp}" / "concatenated_data_cam_mic_sync"
        key = (row.exp, row.file_num)
        if key not in drift_cache:
            wav = audio_path(datadir, args.channel, row.file_num)
            cap = cv2.VideoCapture(str(datadir / row.video))
            with sf.SoundFile(str(wav)) as handle:
                audio_dur = handle.frames / handle.samplerate
            drift_cache[key] = audio_dur / max(video_duration(cap), 1e-9)
            cap.release()

        entry, out = float(row.t_entry), float(row.t_out)
        reverse = direction == "to_nest"
        if reverse:
            n_tiles = strip.shape[1] // TILE_W
            strip = cv2.hconcat([strip[:, i * TILE_W:(i + 1) * TILE_W]
                                 for i in range(n_tiles)][::-1])
        width = strip.shape[1]
        t0 = entry - BEFORE_S
        t1 = t0 + width / PX_PER_S

        audio, fs = read_window(audio_path(datadir, args.channel, row.file_num),
                                t0, t1, drift_cache[key], 0.0)
        spec = spectrogram_tile(audio, fs, width, reverse)
        annotate_spectrogram(spec, t0, t1, entry, out, reverse)
        mark_crossing(spec, t0, t1, entry, out, BEFORE_S, ("enters", "out of tunnel"), reverse)
        y = int(spec.shape[0] * (1 - (GUIDE_KHZ * 1000 - 500) / (45000 - 500)))
        cv2.line(spec, (0, y), (width, y), (110, 110, 110), 1)
        axis = time_axis(width, t0, t1, entry, reverse)
        if row.exp not in das_cache:
            das_cache[row.exp] = all_calls(int(row.exp))
        under = das_cache[row.exp]["underground"].get(row.file_num, [])
        arena = das_cache[row.exp]["arena_1"].get(row.file_num, [])
        bars = [ribbon(under, t0, t1, width, reverse, "DAS underground"),
                ribbon(arena, t0, t1, width, reverse, "DAS arena_1")]

        ok, buf = cv2.imencode(".jpg", cv2.vconcat([spec, *bars, strip, axis]),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            continue
        text = (f"exp {row.exp}  {row.video}  t={entry:.2f}s  {direction}  "
                f"traverse {row.traverse_s:.1f}s  in tunnel {out - entry:.1f}s  "
                f"{'calls: %d tunnel-localised' % row.n_tunnel_calls if row.n_tunnel_calls else 'no tunnel calls'}")
        cards.append((f"{direction}_{label}", f"{row.exp}|{row.video}|{entry:.2f}",
                      "data:image/jpeg;base64," + base64.b64encode(buf).decode(), text))

    path = write_html(cards, Path(args.out_dir))
    print(f"\n{len(cards)} cards -> {path}")


if __name__ == "__main__":
    main()
