"""Show calls assigned to a location where the cameras saw nobody there.

In exp 492, 43% of the calls assigned to arena_2 fall in minutes with zero
detections in arena_2 (against 3% for arena_1). Either the detector is missing
animals, or those calls are being mis-assigned by the audio. One page per call,
so you can tell which by looking:

    row 1   video frame of BOTH arenas at the call's timestamp (annotated
            videos, so a drawn box = the detector saw an animal there)
    row 2   spectrogram of BOTH arena channels around the call, the call's own
            interval shaded

If a gerbil is plainly visible in arena_2 with no box, that is a detection miss.
If arena_2 is empty and the call is far louder on channel 10, the assignment is
wrong. If both arenas are empty, the caller was somewhere the cameras do not
film.

Needs ffmpeg for frame extraction:  module load ffmpeg

    python scripts/analysis/qc_orphan_calls.py --exp 492 --location arena_2
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts" / "utils") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))

from spectrogram_viz import plot_spectrogram  # noqa: E402

from scripts.pipeline.paths import AUDIO_ROOT, video_detections_dir  # noqa: E402
from scripts.pipeline.pool_calls import add_exp_times  # noqa: E402
from scripts.pipeline.pool_detections import CAMERA_TO_LOCATION, FPS, read_files_vetted  # noqa: E402

LOCATION_TO_CAMERA = {v: k for k, v in CAMERA_TO_LOCATION.items()}
LOCATION_TO_CHANNEL = {"arena_1": "10", "arena_2": "20", "underground": "30"}
ARENAS = ["arena_1", "arena_2"]


def find_orphan_calls(date_folder: str, exp: int, location: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calls assigned to `location` in minutes with zero detections there."""
    det = pd.read_parquet(video_detections_dir(date_folder, exp) / "detections.parquet")
    vet = read_files_vetted(video_detections_dir(date_folder, exp) / "files_vetted.csv")
    calls = add_exp_times(exp)

    det["min"] = det.start_time_real.dt.floor("1min")
    per_min = det.groupby(["location", "min"]).size().rename("n_det").reset_index()

    filmed = pd.DataFrame(
        [{"location": r.location, "min": r.chunk_start_real.floor("1min") + pd.Timedelta(minutes=m)}
         for _, r in vet.iterrows() for m in range(6)]
    ).drop_duplicates()
    occ = filmed.merge(per_min, on=["location", "min"], how="left").fillna({"n_det": 0})
    empty_minutes = set(occ.loc[(occ.location == location) & (occ.n_det == 0), "min"])

    calls["min"] = calls.start_time_real.dt.floor("1min")
    mine = calls[(calls.assigned_location == location) & (calls["min"].isin(empty_minutes))].copy()
    return mine.sort_values("start_time_real").reset_index(drop=True), det


def grab_frame(video_path: Path, t_s: float, downscale: int = 2) -> np.ndarray | None:
    """One frame at t_s seconds, via ffmpeg piped as PNG.

    Returned downscaled and as uint8: a full 1600x1200 float32 frame is 23 MB,
    and two of them per page made the PDF 17 MB a page.
    """
    if not video_path.exists():
        return None
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t_s:.3f}",
           "-i", str(video_path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None
    frame = mpimg.imread(io.BytesIO(out.stdout))
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame[::downscale, ::downscale]


def render_call(call, exp: int, date_folder: str, det: pd.DataFrame, window_s: float):
    file_num = int(call.file_num)
    t_call = float(call.start_time_file_sec)
    t0, t1 = max(0.0, t_call - window_s), t_call + window_s

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    vid_dir = video_detections_dir(date_folder, exp)
    audio_dir = AUDIO_ROOT / date_folder / str(exp) / "Averaged_wavs_w_annotations"

    for col, loc in enumerate(ARENAS):
        # --- video frame
        ax = axes[0, col]
        frame = grab_frame(vid_dir / f"video_{LOCATION_TO_CAMERA[loc]}_{file_num:03d}.mp4", t_call)
        n_here = int(((det.location == loc) &
                      (det.start_time_real >= call.start_time_real - pd.Timedelta(seconds=window_s)) &
                      (det.start_time_real <= call.start_time_real + pd.Timedelta(seconds=window_s))).sum())
        if frame is None:
            ax.text(0.5, 0.5, "frame unavailable", ha="center", va="center")
        else:
            ax.imshow(frame)
        ax.set_xticks([]); ax.set_yticks([])
        flag = "  <-- CALL ASSIGNED HERE" if loc == call.assigned_location else ""
        ax.set_title(f"{loc} ({LOCATION_TO_CAMERA[loc]}) — {n_here} detections within ±{window_s:g}s{flag}",
                     fontsize=10, color=("crimson" if loc == call.assigned_location else "black"))

        # --- spectrogram
        ax = axes[1, col]
        wav = audio_dir / f"channel_{LOCATION_TO_CHANNEL[loc]}_file_{file_num:03d}.wav"
        if wav.exists():
            # mmap + slice: these wavs are 360 s at 125 kHz (180 MB), and we need ~2 s
            fs, raw = wavfile.read(wav, mmap=True)
            i0, i1 = int(t0 * fs), min(int(t1 * fs), len(raw))
            x = np.asarray(raw[i0:i1], dtype=np.float32)
            mesh = plot_spectrogram(ax, x, fs, t_start=t0)
            # ~500k quads per panel otherwise land in the PDF as vectors: 17 MB
            # and 35 s per page. Rasterised they are a small image.
            mesh.set_rasterized(True)
            ax.axvspan(t_call, float(call.stop_time_file_sec), color="crimson", alpha=0.18)
            ax.axvline(t_call, color="crimson", lw=0.8)
        else:
            ax.text(0.5, 0.5, f"{wav.name} missing", ha="center", va="center")
        ax.set_title(f"channel {LOCATION_TO_CHANNEL[loc]} ({loc})", fontsize=10)

    fig.suptitle(
        f"exp {exp} · file {file_num} · {call.start_time_real} · {call.event_type} "
        f"· assigned {call.assigned_location} · dur {float(call.duration_sec)*1000:.0f} ms",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date-folder", default="2026_02")
    p.add_argument("--exp", type=int, default=492)
    p.add_argument("--location", default="arena_2")
    p.add_argument("--window-s", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="Only the first N calls (for a quick look).")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "exports")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH. Run:  module load ffmpeg")

    calls, det = find_orphan_calls(args.date_folder, args.exp, args.location)
    if args.limit:
        calls = calls.head(args.limit)
    print(f"exp {args.exp}: {len(calls)} calls assigned to {args.location} in minutes with no "
          f"detections there")
    if calls.empty:
        return 0
    print("  types:", dict(calls.event_type.value_counts()))

    args.out.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out / f"orphan_calls_{args.date_folder}_exp{args.exp}_{args.location}.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, call in enumerate(calls.itertuples(), 1):
            fig = render_call(call, args.exp, args.date_folder, det, args.window_s)
            pdf.savefig(fig, dpi=110); plt.close(fig)
            if i % 10 == 0 or i == len(calls):
                print(f"  {i}/{len(calls)} pages")
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
