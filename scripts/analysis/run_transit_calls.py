#!/usr/bin/env python
"""Do gerbils call when they enter or leave the nest? A PSTH of calls at burrow transits.

Takes the tunnel-crossing events found by the tracking repo's
``burrow_transits.py`` (motion in the burrow_side camera's tunnel ROI; see its
BURROW_HANDOFF.md) and asks whether DAS-detected calls cluster around the moment
an animal crosses -- separately for animals going UP to the arena and DOWN to
the nest, and separately for calls placed underground versus in the arena.

What is compared against what
-----------------------------
Observed: call times relative to the start of each crossing, binned, divided by
the number of crossings and the bin width -> calls/s per crossing.

Baseline: the same computation with each crossing's start time redrawn
uniformly at random inside its own file, 1000 times. Redrawing *within the file*
keeps each file's own call density, so a file that happens to be noisy or busy
cannot manufacture a peak. The grey band is the 2.5th-97.5th percentile of those
shuffles; the observed line leaving the band at some lag is the result.

Time bases
----------
Transit times are VIDEO seconds into a clip; call times in ``calls.csv`` are
AUDIO seconds into the paired wav. The two clocks differ slightly, so transits
are converted with ``drift = audio_duration / video_duration`` measured per file
pair -- the same correction ``sync_video_spectrogram.py`` makes. On experiment
492 the ratio is 1.0000025 (under a millisecond over 360 s); on other
experiments it has reached 0.07%, which is a quarter of a second.

Caveats worth carrying into any conclusion
------------------------------------------
* ``assigned_location`` comes from which mic pair was loudest, not from tracking
  the caller -- an animal in the tunnel is near the underground pair either way,
  so "underground" here means "near the nest end", not "definitely inside".
* Playbacks from the colony speakers are still detected as calls (they have not
  been excluded yet), so a peak that lines up with playback times is suspect.
* Curated transits are the hand-checked subset, so they are the crossings that
  were *visible*; quiet crossings that the motion detector missed are not here.

Usage
-----
    python scripts/analysis/run_transit_calls.py \
        --transits .../transits_492_curated.csv --exp 492 --out-dir exports/transit_calls_492
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from scripts.pipeline.paths import BASE_RAW, experiment_audio_dir  # noqa: E402

DIRECTION_COLORS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}
# validated with the dataviz palette checker: adjacent ΔE 25.2 under protanopia,
# 31.3 under normal vision, both above the surface-contrast floor
BASELINE_GREY = "#b9b9b6"
CALL_TYPES = ["high-freq", "warble", "alarm", "stacks", "newborn"]
N_SHUFFLES = 1000


def read_transits(path: Path) -> list[dict]:
    """Crossing events, each tagged with the file index that pairs it to audio."""
    events = []
    for row in csv.DictReader(open(path)):
        match = re.search(r"_(\d+)\.mp4$", row["video"])
        if not match:
            raise ValueError(f"cannot read a file index out of {row['video']!r}")
        events.append({
            "file_num": int(match.group(1)),
            "video": row["video"],
            "start_s": float(row["start_s"]),
            "end_s": float(row["end_s"]),
            "direction": row["direction"],
        })
    return events


def read_calls(exp: int) -> list[dict]:
    """DAS calls for one experiment, in audio-file seconds."""
    path = experiment_audio_dir(exp) / "calls.csv"
    calls = []
    for row in csv.DictReader(open(path)):
        calls.append({
            "file_num": int(row["file_num"]),
            "start_s": float(row["start_time_file_sec"]),
            "location": row["assigned_location"],
            "event_type": row["event_type"],
        })
    return calls


def measure_drift(datadir: Path, file_nums: set[int], camera: str) -> dict[int, tuple[float, float]]:
    """Per file index: (audio_duration / video_duration, audio_duration).

    Falls back to a ratio of 1.0 when a file pair is incomplete, so one missing
    clip does not sink the whole run.
    """
    out = {}
    for num in sorted(file_nums):
        wav = next((p for p in [datadir / f"channel_01_file_{num:03d}.wav",
                                datadir / f"channel_1_{num}.wav"] if p.is_file()), None)
        video = next((p for p in [datadir / f"video_{camera}_{num:03d}.mp4",
                                  datadir / f"video_{camera}_{num}.mp4"] if p.is_file()), None)
        if wav is None:
            print(f"file {num}: no tunnel wav found, skipping")
            continue
        with sf.SoundFile(str(wav)) as handle:
            audio_dur = handle.frames / handle.samplerate
        ratio = 1.0
        if video is not None:
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            video_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            if video_dur > 0:
                ratio = audio_dur / video_dur
        out[num] = (ratio, audio_dur)
    return out


def calls_by_file(calls: list[dict], location: str | None, event_type: str | None) -> dict[int, np.ndarray]:
    """Call onset times per file index, optionally filtered to one location/type."""
    grouped = defaultdict(list)
    for call in calls:
        if location is not None and call["location"] != location:
            continue
        if event_type is not None and call["event_type"] != event_type:
            continue
        grouped[call["file_num"]].append(call["start_s"])
    return {num: np.sort(np.asarray(times)) for num, times in grouped.items()}


def psth(anchors: list[tuple[int, float]], times: dict[int, np.ndarray],
         edges: np.ndarray) -> np.ndarray:
    """Total call counts per lag bin, summed over anchors (file_num, anchor_time)."""
    counts = np.zeros(len(edges) - 1)
    for file_num, anchor in anchors:
        file_times = times.get(file_num)
        if file_times is None or file_times.size == 0:
            continue
        lags = file_times - anchor
        lags = lags[(lags >= edges[0]) & (lags < edges[-1])]
        if lags.size:
            counts += np.histogram(lags, bins=edges)[0]
    return counts


def shuffle_band(anchors: list[tuple[int, float]], times: dict[int, np.ndarray],
                 edges: np.ndarray, durations: dict[int, float], rng) -> tuple[np.ndarray, np.ndarray]:
    """2.5th / 97.5th percentile of the PSTH when anchors are redrawn in-file."""
    draws = np.zeros((N_SHUFFLES, len(edges) - 1))
    for i in range(N_SHUFFLES):
        fake = [(num, rng.uniform(0.0, durations.get(num, 360.0))) for num, _ in anchors]
        draws[i] = psth(fake, times, edges)
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def rate(counts: np.ndarray, n_anchors: int, bin_s: float) -> np.ndarray:
    """Counts -> calls per second per crossing."""
    return counts / max(n_anchors, 1) / bin_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transits", required=True, help="events CSV from burrow_transits.py")
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=float, default=20.0, help="lag range each side, seconds")
    parser.add_argument("--bin", type=float, default=2.0, help="lag bin width, seconds")
    parser.add_argument("--camera", default="burrow_side")
    parser.add_argument("--datadir", help="override the concatenated_data_cam_mic_sync folder")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")

    transits = read_transits(Path(args.transits))
    calls = read_calls(args.exp)
    drift = measure_drift(datadir, {t["file_num"] for t in transits}, args.camera)
    durations = {num: dur for num, (_, dur) in drift.items()}

    # crossings, converted onto the audio clock so calls and transits share an axis
    anchors_by_direction: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for transit in transits:
        ratio = drift.get(transit["file_num"], (1.0, 360.0))[0]
        anchors_by_direction[("start", transit["direction"])].append(
            (transit["file_num"], transit["start_s"] * ratio))
        anchors_by_direction[("end", transit["direction"])].append(
            (transit["file_num"], transit["end_s"] * ratio))

    edges = np.arange(-args.window, args.window + args.bin, args.bin)
    centers = edges[:-1] + args.bin / 2
    rng = np.random.default_rng(0)
    directions = [d for d in ("to_arena", "to_nest") if anchors_by_direction[("start", d)]]
    locations = ["underground", "arena_1"]

    # Aligning to the crossing START asks "does the animal call as it sets off";
    # aligning to the END asks "does it call on arrival". Crossing durations vary
    # a lot (median ~5 s, but up to 30 s), so one alignment smears whatever the
    # other would resolve -- both are plotted.
    alignments = [("start", "start of the crossing"), ("end", "end of the crossing")]

    psth_rows, summary_rows = [], []
    for align, align_label in alignments:
        fig, axes = plt.subplots(len(directions), len(locations),
                                 figsize=(5.6 * len(locations), 3.4 * len(directions)),
                                 sharex=True, sharey="col", squeeze=False)
        for row, direction in enumerate(directions):
            anchors = anchors_by_direction[(align, direction)]
            for col, location in enumerate(locations):
                ax = axes[row][col]
                times = calls_by_file(calls, location, None)
                observed = rate(psth(anchors, times, edges), len(anchors), args.bin)
                lo_counts, hi_counts = shuffle_band(anchors, times, edges, durations, rng)
                lo = rate(lo_counts, len(anchors), args.bin)
                hi = rate(hi_counts, len(anchors), args.bin)

                ax.fill_between(centers, lo, hi, color=BASELINE_GREY, alpha=0.5,
                                linewidth=0, zorder=1)
                ax.plot(centers, observed, color=DIRECTION_COLORS[direction],
                        linewidth=2, zorder=3)
                ax.axvline(0, color="0.25", linewidth=1, zorder=2)
                ax.set_title(f"{direction} (n={len(anchors)})  -  calls placed {location}",
                             loc="left", fontsize=10)
                ax.grid(axis="y", color="0.92", linewidth=0.8)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if row == len(directions) - 1:
                    ax.set_xlabel(f"seconds from the {align_label}")
                if col == 0:
                    ax.set_ylabel("calls / s / crossing")

                for i, center in enumerate(centers):
                    psth_rows.append({"align": align, "direction": direction,
                                      "location": location,
                                      "lag_s": round(float(center), 3),
                                      "rate_calls_per_s": round(float(observed[i]), 5),
                                      "chance_lo": round(float(lo[i]), 5),
                                      "chance_hi": round(float(hi[i]), 5),
                                      "n_transits": len(anchors)})

                # headline number: calls in the 5 s straddling the alignment point,
                # against the same shuffle null
                near = (centers >= -2.5) & (centers <= 2.5)
                observed_near = float((observed * near).sum() * args.bin)
                draws = np.zeros(N_SHUFFLES)
                for i in range(N_SHUFFLES):
                    fake = [(num, rng.uniform(0.0, durations.get(num, 360.0)))
                            for num, _ in anchors]
                    draws[i] = rate(psth(fake, times, edges), len(anchors), args.bin)[near].sum() * args.bin
                summary_rows.append({
                    "align": align, "direction": direction, "location": location,
                    "n_transits": len(anchors),
                    "calls_per_crossing_within_2.5s": round(observed_near, 3),
                    "chance_median": round(float(np.median(draws)), 3),
                    "p_two_sided": round(float(2 * min((draws >= observed_near).mean(),
                                                       (draws <= observed_near).mean())), 4)})

        fig.suptitle(f"experiment {args.exp}: calling around tunnel crossings, "
                     f"aligned to the {align_label}  (grey = 95% of within-file shuffles)",
                     x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"transit_call_psth_{align}.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "transit_call_psth.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(psth_rows[0]))
        writer.writeheader()
        writer.writerows(psth_rows)
    with open(out_dir / "transit_call_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---- underground calls broken out by call type -------------------------
    present = [t for t in CALL_TYPES if any(c["event_type"] == t and c["location"] == "underground"
                                            for c in calls)]
    fig, axes = plt.subplots(len(directions), len(present), sharex=True,
                             figsize=(3.1 * len(present), 3.0 * len(directions)), squeeze=False)
    for row, direction in enumerate(directions):
        anchors = anchors_by_direction[("start", direction)]
        for col, event_type in enumerate(present):
            ax = axes[row][col]
            times = calls_by_file(calls, "underground", event_type)
            observed = rate(psth(anchors, times, edges), len(anchors), args.bin)
            ax.plot(centers, observed, color=DIRECTION_COLORS[direction], linewidth=1.6)
            ax.axvline(0, color="0.25", linewidth=1)
            ax.set_title(f"{event_type} - {direction}", loc="left", fontsize=9)
            ax.grid(axis="y", color="0.93", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == len(directions) - 1:
                ax.set_xlabel("lag (s)")
            if col == 0:
                ax.set_ylabel("calls / s / crossing")
    fig.suptitle(f"experiment {args.exp}: underground calls around crossing starts, by call type",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "transit_call_psth_by_type.png", dpi=150)
    plt.close(fig)

    # ---- per-transit table --------------------------------------------------
    underground = calls_by_file(calls, "underground", None)
    with open(out_dir / "per_transit_calls.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "start_s", "end_s", "direction",
                         "n_before_5s", "n_during", "n_after_5s"])
        for transit in transits:
            ratio = drift.get(transit["file_num"], (1.0, 360.0))[0]
            a, b = transit["start_s"] * ratio, transit["end_s"] * ratio
            times = underground.get(transit["file_num"], np.empty(0))
            writer.writerow([transit["video"], transit["start_s"], transit["end_s"],
                             transit["direction"],
                             int(((times >= a - 5) & (times < a)).sum()),
                             int(((times >= a) & (times <= b)).sum()),
                             int(((times > b) & (times <= b + 5)).sum())])

    print(f"wrote 3 figures + 3 CSVs -> {out_dir}")
    for row in summary_rows:
        print(f"  {row['align']:>5} | {row['direction']:<8} | {row['location']:<11} | "
              f"n={row['n_transits']:<3} | {row['calls_per_crossing_within_2.5s']:>5} calls per "
              f"crossing within 2.5 s (chance {row['chance_median']}, p={row['p_two_sided']})")


if __name__ == "__main__":
    main()
