#!/usr/bin/env python
"""Where did a call come from -- the tunnel, or deeper in the nest?

Channels 00 and 01 are both underground; ch01 is at the tunnel (verified: +7.2 dB
during transits, versus +2.9 dB on ch00). So the level difference between them
localises a call along the nest-tunnel axis without needing to see the animal --
which matters because the nest cameras cannot count animals at all: the gerbils
burrow under the bedding.

The method is calibrated rather than assumed. Our own tracks say exactly when the
tunnel was EMPTY, and a call at such a moment must have come from somewhere other
than the tunnel. That gives an empirical nest-origin reference distribution. Calls
made while an animal is in the tunnel are then scored against it.

If transit calls sit on the tunnel-dominant side of the reference, the crossing
animal is plausibly the caller -- including for calls before entry and after exit,
which a "was it inside the tunnel period" test wrongly discards.

    python scripts/analysis/localise_calls.py --scan exports/scan_2026_02/492 \
        --exp 492 --out-dir exports/call_origin_492
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import BASE_RAW  # noqa: E402
from scripts.video.burrow_transit_picker import audio_path, file_index, load_calls  # noqa: E402

BAND = (15000, 45000)   # the call band; below this is movement noise
PAD_S = 0.020           # read either side so the filter has room
TUNNEL_CH, NEST_CH = 1, 0
COLOURS = {"tunnel empty": "#d1642a", "animal in tunnel": "#2f6fd0"}


def level_difference(handles, fs, start: float, stop: float, sos) -> float | None:
    """20*log10(rms on the tunnel mic / rms on its nest-side pair), in dB."""
    a = int((start - PAD_S) * fs)
    n = int((stop - start + 2 * PAD_S) * fs)
    if a < 0 or n < 64:
        return None
    levels = []
    for channel in (TUNNEL_CH, NEST_CH):
        handle = handles[channel]
        handle.seek(a)
        x = handle.read(n, dtype="float32")
        if x.size < 64:
            return None
        x = sosfiltfilt(sos, x)
        core = x[int(PAD_S * fs):x.size - int(PAD_S * fs)]
        if core.size < 16:
            return None
        levels.append(np.sqrt(np.mean(core.astype(np.float64) ** 2)) + 1e-12)
    return float(20 * np.log10(levels[0] / levels[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--near", type=float, default=10.0,
                        help="only score calls within this many seconds of a traverse, plus a "
                             "calibration sample -- scoring every call costs 4x the audio reads "
                             "for calls that no analysis looks at")
    parser.add_argument("--calibration", type=int, default=1500,
                        help="tunnel-empty calls sampled per experiment to set that experiment's "
                             "own threshold. Mic gain and tube acoustics differ between "
                             "recordings, so a dB cut from one experiment means something else "
                             "in another -- it has to be recalibrated, not transplanted.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datadir")
    parser.add_argument("--before", type=float, default=3.0, help="seconds before entry counted as transit")
    parser.add_argument("--after", type=float, default=1.0, help="seconds after leaving counted as transit")
    args = parser.parse_args()

    scan = Path(args.scan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")

    calls = load_calls(args.exp)
    tracks = {p.stem: pd.read_parquet(p) for p in (scan / "tracks").glob("*.parquet")}
    # A local run writes one traverses.csv; the disBatch run writes one file per
    # video, because a task that owns a single video cannot safely share a file.
    traverse_files = sorted(scan.glob("traverses*.csv"))
    if not traverse_files:
        raise SystemExit(f"no traverses*.csv under {scan}")
    traverses = [row for path in traverse_files
                 for row in csv.DictReader(open(path))]

    rows = []
    for stem, track in sorted(tracks.items()):
        index = file_index(stem + ".mp4")
        file_calls = calls.get(index, [])
        if not file_calls:
            continue
        n_animals = track.n_animals.to_numpy()

        # transit windows for this file, from the traverses we already trust
        windows = [(float(t["t_entry"]) - args.before, float(t["t_out"]) + args.after)
                   for t in traverses if t["video"] == stem + ".mp4"
                   and str(t["single_animal"]).lower() == "true"]
        entries = [float(t["t_entry"]) for t in traverses if t["video"] == stem + ".mp4"
                   and str(t["single_animal"]).lower() == "true"]

        try:
            handles = {ch: sf.SoundFile(str(audio_path(datadir, ch, index)))
                       for ch in (TUNNEL_CH, NEST_CH)}
        except FileNotFoundError:
            continue
        fs = handles[TUNNEL_CH].samplerate
        sos = butter(4, BAND, btype="band", fs=fs, output="sos")

        rng = np.random.default_rng(index)
        for start, stop, event_type in file_calls:
            near = any(abs(start - e) <= args.near for e in entries)
            frame = int(start * 30)
            if not near:
                # keep a random slice of the far-away calls: they are the calibration
                # reference, and we need enough of them but not all of them
                if frame >= len(n_animals) or n_animals[frame] != 0:
                    continue
                if rng.random() > args.calibration / max(len(file_calls), 1):
                    continue
            frame = int(start * 30)
            if frame >= len(n_animals):
                continue
            occupancy = int(n_animals[frame])
            if occupancy > 1:
                continue                     # ambiguous: two animals in the tube
            delta = level_difference(handles, fs, start, stop, sos)
            if delta is None:
                continue
            rows.append({
                "file": index, "start_s": round(start, 4), "event_type": event_type,
                "tunnel_db_over_nest": round(delta, 2),
                "state": "animal in tunnel" if occupancy == 1 else "tunnel empty",
                "during_transit": any(a <= start <= b for a, b in windows),
            })
        for handle in handles.values():
            handle.close()
        print(f"{stem}: {sum(1 for r in rows if r['file'] == index)} calls scored", flush=True)

    data = pd.DataFrame(rows)
    data.to_csv(out_dir / "call_origin.csv", index=False)

    empty = data[data.state == "tunnel empty"].tunnel_db_over_nest
    occupied = data[data.state == "animal in tunnel"].tunnel_db_over_nest
    transit = data[data.during_transit].tunnel_db_over_nest

    print(f"\n{len(data)} calls scored  (ch{TUNNEL_CH:02d} minus ch{NEST_CH:02d}, "
          f"{BAND[0]//1000}-{BAND[1]//1000} kHz)")
    for label, series in (("tunnel EMPTY (nest-origin reference)", empty),
                          ("animal in tunnel", occupied),
                          ("  ...of those, during a transit", transit)):
        if len(series):
            print(f"  {label:<38} n={len(series):5d}  median {series.median():+6.2f} dB  "
                  f"IQR {series.quantile(.25):+.2f} to {series.quantile(.75):+.2f}")
    if len(empty) and len(occupied):
        threshold = empty.quantile(0.95)
        print(f"\n  95th percentile of the nest reference: {threshold:+.2f} dB")
        print(f"  calls above it while an animal is in the tunnel: "
              f"{100 * (occupied > threshold).mean():.1f}%  (5% expected by construction)")
        if len(transit):
            print(f"  ... during a transit specifically:               "
                  f"{100 * (transit > threshold).mean():.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(-20, 20, 61)
    for label, series in (("tunnel empty", empty), ("animal in tunnel", occupied)):
        if len(series):
            ax.hist(series, bins=bins, density=True, histtype="step", linewidth=2,
                    color=COLOURS[label], label=f"{label} (n={len(series)})")
    ax.axvline(0, color="0.6", linewidth=1)
    ax.set_xlabel("←  nest mic louder            ch01 − ch00 (dB)            tunnel mic louder  →")
    ax.set_ylabel("density")
    ax.set_title(f"experiment {args.exp}: where calls come from, along the nest–tunnel axis",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "call_origin.png", dpi=150)
    print(f"\nwrote {out_dir}/call_origin.csv and call_origin.png")


if __name__ == "__main__":
    main()
