#!/usr/bin/env python
"""Render a centered-playhead spectrogram mp4 with pitch-shifted audio (no video).

Same look-and-feel as sync_video_spectrogram.py minus the camera panels:
one full-width spectrogram strip with a static Hz-label panel on the left and
scrolling time labels at the bottom; "now" is fixed at the centre with a
white vertical line — past audio to the left, upcoming audio to the right.
The audio is pitch-shifted DOWN (default 10x) so ultrasonic calls become
audible while the timeline stays aligned to the spectrogram.

Reuses the spectrogram-rendering and ffmpeg helpers from
sync_video_spectrogram.py — only the filter graph differs (no video inputs,
no vstack). Audio/video clock-drift correction and the residual
AV_SYNC_OFFSET_S are dropped because there is no video to align to.

Normal use: edit the CONFIG block below (START / DURATION to pick the window,
CHANNEL / FILENUM to pick the WAV) and run with no arguments:

    python scripts/play_spectrogram.py

Any CONFIG value can still be overridden from the CLI for one-off runs:

    python scripts/play_spectrogram.py --start 120 --duration 30
    python scripts/play_spectrogram.py -n 63 -c 4 --duration full
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless render; must precede pyplot import
import soundfile as sf

# Pull spectrogram-rendering and ffmpeg helpers from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_video_spectrogram import (  # noqa: E402
    CENTERLINE_W,
    FPS,
    _atempo_chain,
    _audio_candidates,
    _even,
    _resolve_file,
    _run,
    ffprobe_duration,
    parse_duration,
    render_spec_strip_png,
    render_yaxis_overlay_png,
)

# ============================================================================
# CONFIG  --  edit these, then run the script with no arguments
# ============================================================================
EXP      = 235      # experiment id; baked into DATADIR and the output filename

FILENUM  = 156         # file index: channel_<CH>_<N>.wav
CHANNEL  = 4         # mic channel

START    = 279.2   # window start, seconds into the file
DURATION = 5       # window length in seconds;  None (or "full") = whole file

# --- layout ----------------------------------------------------------------
OUTPUT_W = 1280      # output width in px (must be even for libx264)
STRIP_H  = 360       # output height in px (full spec-strip height)

# --- spectrogram window ----------------------------------------------------
# Static spec, sweeping playhead: the full [start, start+duration] is shown
# at once and a white vertical line moves left -> right at output rate.
SPEC_PLAYHEAD = True     # draw the moving "now" line
SPEC_FMIN   = 100        # display range, low edge (Hz)
SPEC_FMAX   = 60000      # display range, high edge (Hz); audio Nyquist 62500

# --- spectrogram rendering (scipy + matplotlib) ----------------------------
SPEC_NFFT      = 256
SPEC_HOP       = 32
SPEC_WINDOW_FN = "hann"
SPEC_CMAP      = "magma"
SPEC_DRANGE_DB = 60
SPEC_VMAX_DB   = None
SPEC_YTICKS_KHZ = (20, 40, 60)
SPEC_TOP_PAD_HZ = 3000
SPEC_YAXIS_W   = 70
SPEC_XAXIS_H   = 30
SPEC_DPI       = 100
SPEC_TICK_STEP_S = 0.1   # time-label tick spacing in seconds (None = auto)

# --- playback speed --------------------------------------------------------
SPEED    = 1     # 1.0 = normal, 1.5 = 1.5x faster, 0.5 = half speed.
                      # Both audio and the moving spectrogram crop are retimed
                      # together; audio pitch is preserved at the pitch-shifted
                      # level.

# --- audio knob (rarely needs changing) ------------------------------------
PITCH    = 10        # audio pitch-shift divisor (10 -> 62.5 kHz -> 6.25 kHz)

DATADIR  = f"/mnt/home/neurostatslab/ceph/saneslab_data/big_setup/experiment_{EXP}/concatenated_data_cam_mic_sync"
OUTDIR   = "."
# ============================================================================


def build_spec_filter(total_w: int, strip_h: int, yaxis_w: int, data_w: int,
                       pps: float, duration: float,
                       img_idx: int, yax_idx: int, aud_idx: int,
                       audio_fs: int, pitch: int,
                       speed: float, playhead: bool) -> str:
    """ffmpeg filter graph: static spec strip + moving playhead + pitched audio.

    The spec PNG is already exactly data_w × strip_h, so we just pad it with
    yaxis_w of black on the left to reach total_w, overlay the static Hz panel
    on top, then composite a moving white line on top. We generate the line as
    a `color` source and use `overlay` (not drawbox) for the playhead, because
    overlay's x expression IS re-evaluated per frame in ffmpeg 7 while
    drawbox's is not. `t` in the overlay expression is input-track time in
    seconds, so the line moves at pps px/s in the un-retimed stream.
    """
    parts = []

    # Pad the spec into the full strip (yaxis_w of black on the left).
    parts.append(
        f"[{img_idx}:v]pad={total_w}:{strip_h}:x={yaxis_w}:y=0:color=black,"
        f"setsar=1,setpts=PTS-STARTPTS[spec_pad]"
    )
    # Static Hz-label panel overlay on the left margin.
    parts.append(
        f"[spec_pad][{yax_idx}:v]"
        f"overlay=x=0:y=0:shortest=0:repeatlast=1[spec_lab]"
    )
    if playhead:
        parts.append(
            f"color=color=white:size={CENTERLINE_W}x{strip_h}:rate={FPS}:"
            f"duration={duration:g}[playhead]"
        )
        parts.append(
            f"[spec_lab][playhead]overlay=x={yaxis_w}+{pps:.6f}*t:y=0:"
            f"shortest=1[spec]"
        )
    else:
        parts.append("[spec_lab]null[spec]")

    # Retime to playback speed. fps=FPS resamples so the output stays at a
    # standard framerate (drops frames for speed>1, duplicates for speed<1).
    parts.append(f"[spec]setpts=PTS/{speed:.6f},fps={FPS}[outv]")

    # Audio: pitch-shift DOWN by `pitch` (sample-rate manipulation, no
    # algorithmic latency), then atempo back up by pitch*speed so the
    # output duration matches the retimed video and pitch stays at the
    # pitch-shifted level.
    new_rate = int(round(audio_fs / pitch))
    tempo_chain = _atempo_chain(float(pitch * speed))
    parts.append(
        f"[{aud_idx}:a]asetrate={new_rate},aresample=48000,{tempo_chain},"
        f"asetpts=PTS-STARTPTS[outa]"
    )
    return ";".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render a centered-playhead spectrogram mp4 with "
                    "pitch-shifted audio. No video panels.")
    p.add_argument("-n", "--filenum", type=int, default=FILENUM,
                   help=f"file index (default {FILENUM})")
    p.add_argument("-c", "--channel", type=int, default=CHANNEL,
                   help=f"mic channel (default {CHANNEL})")
    p.add_argument("-s", "--start", type=float, default=START,
                   help=f"window start in seconds (default {START})")
    p.add_argument("-t", "--duration", default=DURATION,
                   help=f"window length in seconds, or 'full' (default {DURATION})")
    p.add_argument("-d", "--datadir", default=DATADIR, help="input data directory")
    p.add_argument("-o", "--output", default=None,
                   help="output mp4 path (default: auto-named in OUTDIR)")
    p.add_argument("--speed", type=float, default=SPEED,
                   help=f"output playback speed (default {SPEED}). Audio "
                        "pitch is preserved at the pitch-shifted level.")
    args = p.parse_args()
    if args.speed <= 0:
        print(f"ERROR: --speed must be > 0 (got {args.speed})", file=sys.stderr)
        return 1

    datadir = Path(args.datadir)
    try:
        audio = _resolve_file(datadir, _audio_candidates(args.channel, args.filenum))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # geometry
    total_w = _even(OUTPUT_W)
    strip_h = _even(STRIP_H)
    yaxis_w = _even(SPEC_YAXIS_W)
    xaxis_h = _even(SPEC_XAXIS_H)
    data_w  = total_w - yaxis_w
    if data_w <= 0:
        print(f"ERROR: SPEC_YAXIS_W ({yaxis_w}) >= OUTPUT_W ({total_w})",
              file=sys.stderr)
        return 2

    # Static-spec mode: the full [start, start+duration] is rendered once at
    # exactly data_w pixels wide; no past/future padding needed because the
    # spec doesn't scroll. pps (pixels per source-second) drives the moving
    # playhead's x expression in the filter graph.
    file_dur = ffprobe_duration(audio)
    audio_fs = sf.info(str(audio)).samplerate
    duration = parse_duration(args.duration)
    if duration is None:
        duration = file_dur - args.start
    s0 = args.start
    s1 = args.start + duration
    pps = data_w / duration

    if args.output:
        out = Path(args.output)
    else:
        window = f"{args.start:g}+{duration:g}s"
        speed_tag = f"_{args.speed:g}x" if args.speed != 1.0 else ""
        out = Path(OUTDIR) / (
            f"spec_exp{EXP}_ch{args.channel}_file{args.filenum}"
            f"_{window}{speed_tag}.mp4")

    print(f">>> audio : {audio}  (pitch /{PITCH})")
    print(f">>> window: {args.start:g}s for {duration:g}s "
          f"(static spec, sweeping playhead, "
          f"playback speed {args.speed:g}x = {duration/args.speed:g}s of output)")
    print(f">>> output: {out}  ({total_w}x{strip_h})\n")

    with tempfile.TemporaryDirectory() as tmp:
        spec_png  = Path(tmp) / "spec_strip.png"
        yaxis_png = Path(tmp) / "yaxis.png"

        # pass 1a: spec covering exactly [start, start+duration] at data_w wide.
        # drift_ratio = 1.0 and av_offset = 0.0: no video to align to.
        render_spec_strip_png(audio, s0, s1, file_dur,
                              1.0, 0.0,
                              data_w, strip_h, xaxis_h,
                              SPEC_CMAP, None, SPEC_VMAX_DB, SPEC_DRANGE_DB,
                              SPEC_FMIN, SPEC_FMAX, SPEC_TOP_PAD_HZ,
                              SPEC_NFFT, SPEC_HOP, SPEC_WINDOW_FN,
                              SPEC_DPI, spec_png,
                              label_offset=s0,
                              tick_step_s=SPEC_TICK_STEP_S)
        # pass 1b: static y-axis (Hz) overlay
        render_yaxis_overlay_png(yaxis_w, strip_h, xaxis_h,
                                  SPEC_FMIN, SPEC_FMAX, SPEC_TOP_PAD_HZ,
                                  SPEC_YTICKS_KHZ, SPEC_DPI, yaxis_png)

        # pass 2: static spec + moving playhead + pitched audio
        cmd = ["ffmpeg", "-y", "-hide_banner",
               "-loop", "1", "-framerate", str(FPS),     # spec strip [0]
               "-t", f"{duration:g}", "-i", str(spec_png),
               "-loop", "1", "-framerate", str(FPS),     # yaxis overlay [1]
               "-t", f"{duration:g}", "-i", str(yaxis_png),
               "-ss", f"{args.start:.6f}", "-t", f"{duration:.6f}",  # audio [2]
               "-i", str(audio)]

        filtergraph = build_spec_filter(total_w, strip_h, yaxis_w, data_w,
                                         pps, duration, 0, 1, 2,
                                         audio_fs, PITCH, args.speed,
                                         SPEC_PLAYHEAD)
        cmd += ["-filter_complex", filtergraph,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                str(out)]

        res = _run(cmd)
        if res.returncode != 0:
            print(f"\nffmpeg failed (exit {res.returncode})", file=sys.stderr)
            return res.returncode

    print(f"\nDone -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
