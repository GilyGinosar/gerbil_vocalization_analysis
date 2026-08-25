#!/usr/bin/env python
"""Is the entry peak behaviour, or just relabelling? Transits versus dwelling.

While an animal is in the tunnel, any call it makes is classified tunnel-origin
whether or not it is calling more than usual. Since the animal is in the tunnel
around t=0, that alone would produce a peak in tunnel-origin calls at entry. The
8.5x figure cannot be read as behaviour until that is excluded.

The control: periods when EXACTLY ONE animal is in the tunnel and does NOT cross
-- it pauses, sniffs, turns back. Identical classification mechanics, identical
occupancy, no traverse. If transits call faster than dwells, the peak is
behaviour. If they match, it is bookkeeping.

Note this cannot be answered from the localiser's own output: that scored calls
near traverses plus an empty-tunnel calibration sample, so dwell-period calls
were never scored and their rate would come out near zero by construction. This
does its own scoring pass over both period types, using the threshold the
localiser already established for that experiment.

    python scripts/analysis/dwell_control.py --scan <scan>/492 --exp 492 \
        --origin <scan>/origin/492/call_origin.csv --out <scan>/dwell/492.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import BASE_RAW  # noqa: E402
from scripts.analysis.localise_calls import BAND, NEST_CH, TUNNEL_CH, level_difference  # noqa: E402
from scripts.video.burrow_transit_picker import audio_path, file_index, load_calls  # noqa: E402

MIN_PERIOD_S = 0.5      # ignore flickers
MAX_PERIODS = 400       # sampled per experiment per type, enough for a rate estimate


def single_animal_periods(track: pd.DataFrame) -> list[tuple[float, float]]:
    """Contiguous stretches with exactly one animal in the tunnel."""
    flag = (track.n_animals.to_numpy() == 1).astype(np.int8)
    edges = np.diff(np.concatenate(([0], flag, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(a / 30.0, b / 30.0) for a, b in zip(starts, stops)
            if (b - a) / 30.0 >= MIN_PERIOD_S]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True, help="one experiment's scan dir")
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--origin", required=True, help="that experiment's call_origin.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    origin = pd.read_csv(args.origin)
    reference = origin.loc[origin.state == "tunnel empty", "tunnel_db_over_nest"]
    if len(reference) < 50:
        raise SystemExit(f"exp {args.exp}: only {len(reference)} calibration calls")
    threshold = float(reference.quantile(0.95))

    scan = Path(args.scan)
    datadir = BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync"
    calls = load_calls(args.exp)
    traverses = [r for path in sorted(scan.glob("traverses*.csv"))
                 for r in csv.DictReader(open(path))
                 if str(r["single_animal"]).lower() == "true"]
    by_video: dict[str, list[tuple[float, float]]] = {}
    for row in traverses:
        by_video.setdefault(row["video"], []).append(
            (float(row["t_entry"]), float(row["t_out"])))

    rng = np.random.default_rng(args.exp)
    rows = []
    for path in sorted((scan / "tracks").glob("*.parquet")):
        stem = path.stem
        index = file_index(stem + ".mp4")
        file_calls = calls.get(index, [])
        if not file_calls:
            continue
        track = pd.read_parquet(path)
        crossings = by_video.get(stem + ".mp4", [])

        periods = []
        for start, stop in single_animal_periods(track):
            # a period is a transit if a detected crossing happens inside it
            is_transit = any(start - 0.5 <= a and b <= stop + 0.5 for a, b in crossings)
            periods.append((start, stop, "transit" if is_transit else "dwell"))
        if not periods:
            continue

        try:
            handles = {ch: sf.SoundFile(str(audio_path(datadir, ch, index)))
                       for ch in (TUNNEL_CH, NEST_CH)}
        except FileNotFoundError:
            continue
        fs = handles[TUNNEL_CH].samplerate
        sos = butter(4, BAND, btype="band", fs=fs, output="sos")

        times = np.array([c for c, _, _ in file_calls])
        for start, stop, kind in periods:
            inside = [(s, e, t) for s, e, t in file_calls if start <= s <= stop]
            n_localised = 0
            for call_start, call_stop, _t in inside:
                delta = level_difference(handles, fs, call_start, call_stop, sos)
                if delta is not None and delta > threshold:
                    n_localised += 1
            rows.append({"exp": args.exp, "file": index, "kind": kind,
                         "start_s": round(start, 3), "duration_s": round(stop - start, 3),
                         "n_calls": len(inside), "n_tunnel_calls": n_localised})
        for handle in handles.values():
            handle.close()

    table = pd.DataFrame(rows)
    if table.empty:
        raise SystemExit(f"exp {args.exp}: no periods scored")
    # cap after the fact so both types keep their natural per-file spread
    parts = []
    for kind, group in table.groupby("kind"):
        parts.append(group.sample(min(len(group), MAX_PERIODS), random_state=args.exp))
    table = pd.concat(parts, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    for kind, group in table.groupby("kind"):
        seconds = group.duration_s.sum()
        print(f"  exp {args.exp} {kind:<8} n={len(group):4d}  {seconds:8.1f} s in tunnel  "
              f"{group.n_tunnel_calls.sum():5d} tunnel calls  "
              f"{group.n_tunnel_calls.sum()/max(seconds,1e-9):.3f} /s", flush=True)


if __name__ == "__main__":
    main()
