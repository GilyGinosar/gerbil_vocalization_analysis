#!/usr/bin/env python
"""Was anything moving in the nest before the animal arrived -- and does it matter?

The nest cannot be counted from video (the gerbils burrow under the bedding), but
it can be asked whether anything MOVED, which is a far easier measurement and,
unlike the prior-nest-call split, is independent of the audio. That matters:
splitting on nest CALLS to ask whether the nest was calling is close to circular,
while splitting on nest MOTION is not.

For a `to_nest` traverse the animal is in the arena before entry, so it is not in
the nest camera's view at all -- the pre-entry window is residents only. After it
leaves the tunnel it IS in view, so post-arrival motion is contaminated by the
traveller and is reported but not used for the split.

This is the windowed pilot: it decodes ~10 s per traverse rather than whole files,
which is ~30x less video than a full scan, at the cost of not leaving a reusable
trace behind. Run it first; only build the full scan if the split earns it.

    python scripts/analysis/nest_motion.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/burrow/nest_motion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.raster_and_rate import load_usv  # noqa: E402
from scripts.analysis.raster_and_rate_tunnel import ALL_TYPES  # noqa: E402
from scripts.pipeline.paths import BASE_RAW  # noqa: E402

FPS = 30
SMALL = (200, 150)     # frames are downscaled before differencing; the question is
                       # "did anything move", not where
CHANGED = 12           # a pixel counts as changed at this absolute difference


def motion_window(cap, t0: float, t1: float) -> np.ndarray:
    """Fraction of pixels changing frame to frame, across [t0, t1)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t0 * FPS), 0))
    n = max(int((t1 - t0) * FPS), 1)
    prev, out = None, []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SMALL)
        if prev is not None:
            out.append(float((cv2.absdiff(small, prev) > CHANGED).mean()))
        prev = small
    return np.array(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--pre", type=float, default=5.0, help="seconds before entry")
    parser.add_argument("--post", type=float, default=2.0, help="seconds after leaving")
    parser.add_argument("--limit", type=int, default=400, help="traverses to sample")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = pd.read_parquet(scan / f"traverses_{args.date}.parquet")
    tv = tv[tv.single_animal & (tv.direction == "to_nest")]
    # spread the sample across experiments rather than taking one experiment's worth
    order = tv.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
    tv = tv.assign(rank=order).sort_values(["rank", "exp"]).head(args.limit)
    print(f"{len(tv)} to_nest traverses from {tv.exp.nunique()} experiments")

    calls: dict[int, dict] = {}
    rows = []
    for (exp, file_num), group in tv.groupby(["exp", "file_num"]):
        path = (BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
                / f"video_nest_top_{int(file_num):03d}.mp4")
        if not path.exists():
            continue
        if exp not in calls:
            calls[exp] = load_usv(int(exp), ("underground",), ALL_TYPES)
        train = calls[exp].get(int(file_num), np.empty(0))
        cap = cv2.VideoCapture(str(path))
        for r in group.sort_values("t_entry").itertuples():
            pre = motion_window(cap, r.t_entry - args.pre, r.t_entry)
            post = motion_window(cap, r.t_out, r.t_out + args.post)
            if not pre.size:
                continue
            arrival = int(((train >= r.t_out - 0.5) & (train <= r.t_out + 2.0)).sum())
            rows.append({"exp": int(exp), "file_num": int(file_num),
                         "t_entry": float(r.t_entry), "t_out": float(r.t_out),
                         "in_tunnel": float(r.t_out - r.t_entry),
                         "motion_pre": float(pre.mean()),
                         "motion_pre_max": float(pre.max()),
                         "still_frac": float((pre < 0.005).mean()),
                         "motion_post": float(post.mean()) if post.size else np.nan,
                         "arrival_calls": arrival,
                         "arrival_rate": arrival / 2.5})
        cap.release()
        print(f"  exp {exp} file {file_num}: {len(rows)} done", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "nest_motion.csv", index=False)
    print(f"\n{len(d)} traverses measured\n")
    print("pre-entry nest motion (fraction of pixels changing per frame):")
    print(f"  median {d.motion_pre.median():.4f}   p25 {d.motion_pre.quantile(.25):.4f}"
          f"   p75 {d.motion_pre.quantile(.75):.4f}")
    print(f"  post-arrival median {d.motion_post.median():.4f}  "
          f"(includes the traveller, not used for the split)\n")

    cut = d.motion_pre.median()
    quiet, active = d[d.motion_pre <= cut], d[d.motion_pre > cut]
    print(f"arrival call rate, split at the median pre-entry motion ({cut:.4f}):")
    for name, s in (("nest QUIET before", quiet), ("nest ACTIVE before", active)):
        print(f"  {name:<20} n={len(s):4d}   {s.arrival_rate.mean():.3f} calls/s   "
              f"({100*(s.arrival_calls > 0).mean():.0f}% had any call at arrival)")
    rng = np.random.default_rng(0)
    obs = active.arrival_rate.mean() - quiet.arrival_rate.mean()
    lab = (d.motion_pre > cut).to_numpy()
    vals = d.arrival_rate.to_numpy()
    null = np.array([(lambda m: vals[m].mean() - vals[~m].mean())(rng.permutation(lab))
                     for _ in range(5000)])
    print(f"  difference {obs:+.3f} calls/s   permutation p = "
          f"{(np.abs(null) >= abs(obs)).mean():.4f}")
    print(f"\nwrote {out_dir}/nest_motion.csv")


if __name__ == "__main__":
    main()
