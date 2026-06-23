#!/usr/bin/env python
"""Render a multi-camera video + synced centered-playhead spectrogram mp4.

Output is a single mp4: one or more camera views placed side-by-side, with a
labeled spectrogram strip of one mic channel on top. The spectrogram scrolls
so that "now" is fixed in the MIDDLE of the data area (a white vertical line
marks it) - past audio to the left, upcoming audio to the right. The audio is
recorded at 125 kHz (ultrasonic), so the muxed soundtrack is pitch-shifted
DOWN (default 10x) to make calls audible while keeping the original duration
-> it stays synced to the video.

The spectrogram is computed with scipy.signal.stft and rendered with
matplotlib (cmap=magma by default) for a crisp, axis-labeled strip.

Layout:

    +--------+-----------------------------------------+
    | Hz     |   spectrogram   ( | = now, centered )   |  <- SPEC_FRAC * video_h
    | labels |  ........................................|
    | (left) |   time (s) labels   (scroll with data)  |
    +--------+----------------+------------------------+
    |   VIDEO_VIEWS[0]        |     VIDEO_VIEWS[1] ... |
    +-------------------------+------------------------+

How the centered playhead works: pass 1 pre-renders the strip as one wide
static PNG spanning SPEC_WINDOW/2 seconds before the window to SPEC_WINDOW/2
after it (data area + scrolling time labels). Pass 2 crops a moving
SPEC_WINDOW-wide slice out of it and overlays the static Hz-label panel on
the left, so the data area centre always shows the current audio moment.

Normal use: edit the CONFIG block below (especially START / DURATION to pick the
window, and VIDEO_VIEWS to pick cameras) and run with no arguments:

    python scripts/sync_video_spectrogram.py

Any CONFIG value can still be overridden from the CLI for one-off runs:

    python scripts/sync_video_spectrogram.py --start 120 --duration 30
    python scripts/sync_video_spectrogram.py --views nest_side burrow_side
    python scripts/sync_video_spectrogram.py -n 63 --duration full

Each camera column can carry its own spectrogram, share one stretched across
several columns, or have none. Strips are configured in SPEC_STRIPS (or via
--specs) as 'view[+view...]:channel' -- a strip listing several views is
stretched above all of them; uncovered views get blank (black) space:

    # ch0 above 'center', ch4 stretched above nest_side+burrow_side
    python scripts/sync_video_spectrogram.py --views center nest_side burrow_side \
        --specs center:0 nest_side+burrow_side:4

The muxed (pitch-shifted) soundtrack is always a single channel -- by default
the first strip's channel, or set AUDIO_CHANNEL / --audio-channel.

Available camera views: burrow_side, burrow_top, center, gily_center,
nest_side, nest_top.

ffmpeg is provided by the `ffmpeg/7.1.1-nix` Lmod module (loaded automatically;
no install needed on FI workstations).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless render; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

# ============================================================================
# CONFIG  --  edit these, then run the script with no arguments
# ============================================================================
EXP      = 492       # experiment id; baked into DATADIR and the output filename

FILENUM  = 15        # file index: channel_<CH>_<N>.wav / video_<view>_<N>.mp4

START    = 238      # window start, seconds into the file
DURATION = 130        # window length in seconds;  None (or "full") = swhole file

# --- layout ----------------------------------------------------------------
VIDEO_VIEWS = ["center","burrow_side","nest_side"]  # camera views, placed left -> right

# Each spectrogram strip sits above a CONTIGUOUS group of VIDEO_VIEWS columns.
# `views` must be a contiguous left->right slice of VIDEO_VIEWS; a strip listing
# more than one view is STRETCHED across all of them. Any view not covered by a
# strip gets blank (black) space above it. A single strip covering every view
# reproduces the old one-spectrogram behaviour.
SPEC_STRIPS = [
    {"channel": 0, "views": ["center"]},
    {"channel": 4, "views": ["burrow_side","nest_side"]},
]
AUDIO_CHANNEL = None   # mic channel for the muxed (pitched) soundtrack;
                       # None -> first SPEC_STRIPS channel

VIDEO_FLIP  = {}#{"burrow_side": "v"}   # per-view flip: "v" (vertical), "h", or "hv"
VIDEO_W     = 720    # width of EACH video panel in px (video is 4:3)
SPEC_FRAC   = 0.4    # spectrogram strip height as a fraction of the video height

# --- spectrogram --------------------------------------------------------------
SPEC_WINDOW    = 2      # seconds of audio visible across the strip ("now" is
                         # centered: SPEC_WINDOW/2 of past + SPEC_WINDOW/2 ahead)
SPEC_CENTERLINE = True   # draw the white "now" line down the middle
SPEC_FMIN   = 100    # spectrogram display range, low edge (Hz)
SPEC_FMAX   = 60000  # spectrogram display range, high edge (Hz); audio Nyquist 62500

# --- spectrogram rendering (scipy + matplotlib) ----------------------------
SPEC_NFFT      = 2048    # STFT window size in samples (Δf at 125 kHz ≈ 61 Hz)
SPEC_HOP       = 256     # STFT hop length in samples  (Δt at 125 kHz ≈ 2 ms)
SPEC_WINDOW_FN = "hann"  # STFT window function
SPEC_CMAP      = "magma" # matplotlib colormap
SPEC_DRANGE_DB = 80      # displayed dynamic range in dB; widening past the noise
                          # floor (~60 dB) lifts the background into magma's dark
                          # purple range instead of pure black -> "vivid" look
SPEC_VMAX_DB   = None    # dB ceiling; None = auto (peak of |S|_db in window)
SPEC_YTICKS_KHZ = (20, 40, 60)   # Hz tick labels to show on the static y-axis (kHz)
SPEC_TOP_PAD_HZ = 3000   # blank Hz above SPEC_FMAX so the topmost tick label
                          # doesn't clip against the figure's top edge
SPEC_YAXIS_W   = 70      # px width of the static Hz-label panel on the left
SPEC_XAXIS_H   = 30      # px height of the scrolling time-label strip at bottom
SPEC_DPI       = 100     # matplotlib DPI when sizing figures to exact px dims

# --- playback speed -------------------------------------------------------
SPEED    = 1      # output playback speed: 1.0 = normal, 1.5 = 1.5x faster,
                      # 0.5 = half speed. Applies to BOTH video and audio; audio
                      # pitch is preserved (so the pitch-shifted calls stay at
                      # the same key, just play faster). Useful for scanning
                      # long recordings. Override with --speed N.

# --- audio knobs (rarely need changing) ------------------------------------
PITCH    = 10        # audio pitch-shift divisor (10 -> 62.5 kHz call -> 6.25 kHz)
AV_SYNC_OFFSET_S = 0.15  # extra constant offset (seconds) added to every audio
                          # read AFTER the automatic clock-drift correction.
                          # Drift between the audio (nominal 125 kHz) and video
                          # (nominal 30 fps) clocks is auto-computed from each
                          # file pair's reported durations; the residual 0.15 s
                          # was tuned empirically against gerbil foot impacts on
                          # experiment_237/file63 (likely the audio recording
                          # started ~150 ms before the video). Tune with --av-offset.

DATADIR  = f"/mnt/home/neurostatslab/ceph/saneslab_data/big_setup/experiment_{EXP}/concatenated_data_cam_mic_sync"
OUTDIR   = OUTDIR = "/mnt/home/gginosar/repos/gerbil_vocalization_analysis" #"."       # directory for the output mp4 (use ceph for full renders)
# ============================================================================

FFMPEG_MODULE = "ffmpeg/7.1.1-nix"
VIDEO_ASPECT = 1200 / 1600  # source cameras are 1600x1200
FPS = 30
CENTERLINE_W = 3            # white "now" line width in px


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command with the ffmpeg module loaded.

    The command and its args are passed through bash's positional parameters
    (``"$@"``), so there is no shell-quoting hazard regardless of the args.
    """
    wrapper = (
        "source /etc/profile.d/modules.sh 2>/dev/null || true; "
        f"module load {FFMPEG_MODULE} >/dev/null 2>&1 && exec \"$@\""
    )
    return subprocess.run(["bash", "-c", wrapper, "_", *argv], **kw)


def _even(x: float) -> int:
    """Round to the nearest even integer (libx264 needs even dimensions)."""
    return int(round(x / 2)) * 2


_FLIP_FILTERS = {"v": "vflip", "h": "hflip", "hv": "hflip,vflip"}


def flip_filter(code: str) -> str:
    """Map a flip code ('v', 'h', 'hv') to ffmpeg filter(s); '' for no flip."""
    if not code:
        return ""
    key = "".join(sorted(code.lower()))
    if key not in _FLIP_FILTERS:
        raise ValueError(f"bad flip code {code!r}; use 'v', 'h', or 'hv'")
    return _FLIP_FILTERS[key]


def ffprobe_duration(path: Path) -> float:
    """Return the duration (seconds) of a media file."""
    res = _run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(res.stdout.strip())


def _resolve_file(datadir: Path, name_candidates: list[str]) -> Path:
    """Return the first ``datadir / name`` from ``name_candidates`` that exists.

    Raises ``FileNotFoundError`` listing every (de-duplicated) name tried so
    the user can see exactly which naming conventions were probed.
    """
    tried = []
    for name in name_candidates:
        if name in tried:
            continue
        tried.append(name)
        path = datadir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no file in {datadir} matching any of: {', '.join(tried)}"
    )


def _audio_candidates(channel: int, num: int) -> list[str]:
    """Audio filenames across the conventions used by different experiment dumps.

    Two patterns are known in the wild and we try all zero-pad combinations:

        channel_<C>_<N>.wav              (e.g. exp_237: channel_4_63.wav)
        channel_<CC>_file_<NNN>.wav      (e.g. exp_518: channel_01_file_012.wav)

    Returns the cross-product of {1, 2}-wide channel padding, {"_",
    "_file_"} infix, and {1..4}-wide file-number padding.
    """
    return [
        f"channel_{channel:0{cw}d}{ix}{num:0{nw}d}.wav"
        for cw in (1, 2)
        for ix in ("_", "_file_")
        for nw in (1, 2, 3, 4)
    ]


def _video_candidates(view: str, num: int) -> list[str]:
    """Video filenames across the zero-pad conventions used by different dumps.

    Pattern is ``video_<view>_<N>.mp4`` in every dump observed so far; the
    only difference is the file-number padding width (e.g. ``..._63.mp4`` vs
    ``..._012.mp4``).
    """
    return [f"video_{view}_{num:0{w}d}.mp4" for w in (1, 2, 3, 4)]


def _load_audio_window(audio: Path, s0: float, s1: float, file_dur: float,
                       drift_ratio: float = 1.0, av_offset_s: float = 0.0,
                       ) -> tuple[np.ndarray, float]:
    """Return mono float32 audio for VIDEO-time window [s0, s1].

    The audio FILE is read from [s0*drift_ratio + offset, s1*drift_ratio +
    offset] (audio-file time) so the returned samples correspond exactly to
    the video-time interval [s0, s1] -- compensating for clock-drift between
    the audio (nominal 125 kHz) and video (nominal 30 fps) sample clocks.

    The returned ``fs`` is the CORRECTED sample rate (``nominal_fs *
    drift_ratio``); passed to scipy.signal.stft this makes ``t_axis`` come
    out in video-time seconds directly, so plotting at ``t_axis + s0`` puts
    each STFT column at the correct video-time position on the strip.

    Padding both ends keeps the spectrogram image's time axis exactly aligned
    to [s0, s1] even when the requested window runs past the start/end of the
    recording -- which is what the moving-crop math in pass 2 relies on.
    """
    s0_file = s0 * drift_ratio + av_offset_s
    s1_file = s1 * drift_ratio + av_offset_s
    s0c = max(0.0, s0_file)
    s1c = min(file_dur, s1_file)
    info = sf.info(str(audio))
    nominal_fs = info.samplerate
    start_frame = int(round(s0c * nominal_fs))
    stop_frame = int(round(s1c * nominal_fs))
    data, _ = sf.read(str(audio), start=start_frame, stop=stop_frame,
                       dtype="float32")
    if data.ndim > 1:                       # mic 0 if multichannel
        data = data[:, 0]
    pre = int(round(max(0.0, s0c - s0_file) * nominal_fs))
    post = int(round(max(0.0, s1_file - s1c) * nominal_fs))
    if pre or post:
        data = np.concatenate([np.zeros(pre, dtype=data.dtype),
                                data,
                                np.zeros(post, dtype=data.dtype)])
    return data, nominal_fs * drift_ratio


def render_spec_strip_png(audio: Path, s0: float, s1: float, file_dur: float,
                          drift_ratio: float, av_offset_s: float,
                          full_w: int, strip_h: int, xaxis_h: int,
                          cmap: str, vmin_db: float | None, vmax_db: float | None,
                          drange_db: float, fmin: float, fmax: float,
                          top_pad_hz: float,
                          nfft: int, hop: int, win_fn: str,
                          dpi: int, out_png: Path,
                          label_offset: float | None = None,
                          tick_step_s: float | None = None) -> None:
    """Pass 1a: render the wide spectrogram strip as a single PNG.

    The PNG is exactly (full_w × strip_h) pixels. The top (strip_h - xaxis_h)
    px are the spectrogram data (scipy STFT, log-power dB, ``cmap`` colormap)
    spanning [s0, s1] in time and [fmin, fmax] in frequency. The bottom
    ``xaxis_h`` px carry the time-axis tick labels. The y-axis Hz labels are
    NOT drawn here -- a separate static overlay supplies them so they stay put
    when the strip scrolls in pass 2.
    """
    # Load audio for VIDEO-time window [s0, s1], compensating for audio/video
    # clock drift via drift_ratio and an extra constant av_offset_s. The
    # returned fs is the CORRECTED rate so scipy's t_axis comes out in
    # video-time seconds and aligns with the strip's labeled time axis below.
    data, fs = _load_audio_window(audio, s0, s1, file_dur,
                                   drift_ratio=drift_ratio,
                                   av_offset_s=av_offset_s)
    f_axis, t_axis, Z = stft(data, fs=fs, nperseg=nfft, noverlap=nfft - hop,
                              window=win_fn, padded=False, boundary=None)
    S_db = 20.0 * np.log10(np.abs(Z) + 1e-10)
    # Mask out frequencies above fmax so the small empty band above (added by
    # top_pad_hz to give the topmost tick label room) stays empty / black
    # instead of showing near-Nyquist noise the user didn't ask to see.
    mask = f_axis <= fmax
    f_axis = f_axis[mask]
    S_db = S_db[mask, :]
    if vmax_db is None:
        vmax_db = float(np.max(S_db))
    if vmin_db is None:
        vmin_db = vmax_db - drange_db

    fig = plt.figure(figsize=(full_w / dpi, strip_h / dpi), dpi=dpi,
                     facecolor="black")
    spec_h = strip_h - xaxis_h

    # Data axes: top portion, full width, no margins -> pixels span [s0,s1]
    ax = fig.add_axes((0.0, xaxis_h / strip_h, 1.0, spec_h / strip_h))
    ax.pcolormesh(t_axis + s0, f_axis, S_db, shading="auto",
                  cmap=cmap, vmin=vmin_db, vmax=vmax_db, rasterized=True)
    ax.set_xlim(s0, s1)
    ax.set_ylim(fmin, fmax + top_pad_hz)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Time-label strip: thin band along the bottom of the figure. The axes
    # box's bottom spine carries the tick marks (pointing up, into the band);
    # labels render in the small pad BELOW the spine. We reserve `label_pad`
    # px below the axes box so those labels stay inside the figure -- if
    # bottom == 0, matplotlib draws the labels below the figure and they get
    # clipped to black.
    label_pad = 14    # px reserved for tick-label text below the tick line
    band_h = max(2, xaxis_h - label_pad)
    ax_t = fig.add_axes((0.0, label_pad / strip_h, 1.0, band_h / strip_h))
    ax_t.set_xlim(s0, s1)
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([])
    ax_t.set_facecolor("black")
    # Tick spacing: caller can override via tick_step_s (e.g. 0.1 for 100 ms
    # ticks); otherwise auto-pick an integer-second cadence targeting ~one
    # label every ~80 px.
    if tick_step_s is None:
        px_per_label_target = 80
        target_n = max(2, int(round(full_w / px_per_label_target)))
        tick_step = max(1, int(round((s1 - s0) / target_n)))
    else:
        tick_step = float(tick_step_s)

    # Label format: integer for whole-second steps, fixed-decimal for sub-second.
    if tick_step >= 1.0 and abs(tick_step - round(tick_step)) < 1e-9:
        fmt = lambda v: f"{int(round(v))}"
    else:
        decimals = max(1, -int(np.floor(np.log10(tick_step))))
        fmt = lambda v: f"{v:.{decimals}f}"

    if label_offset is None:
        # Default: integer-file-time ticks, labels show the file-time value.
        ticks = np.arange(np.ceil(s0), np.floor(s1) + 1, tick_step)
        labels = [fmt(t) for t in ticks]
    else:
        # Ticks anchored to label_offset; labels are seconds from it.
        k_min = int(np.ceil((s0 - label_offset) / tick_step))
        k_max = int(np.floor((s1 - label_offset) / tick_step))
        ks = np.arange(k_min, k_max + 1)
        ticks = label_offset + ks * tick_step
        labels = [fmt(k * tick_step) for k in ks]
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels(labels, color="white", fontsize=9)
    ax_t.tick_params(axis="x", colors="white", direction="in", length=4,
                     pad=2, top=False, bottom=True,
                     labeltop=False, labelbottom=True)
    for side in ("top", "left", "right"):
        ax_t.spines[side].set_visible(False)
    ax_t.spines["bottom"].set_color("white")
    ax_t.spines["bottom"].set_linewidth(0.5)

    fig.savefig(out_png, dpi=dpi, facecolor="black")
    plt.close(fig)


def render_yaxis_overlay_png(yaxis_w: int, strip_h: int, xaxis_h: int,
                              fmin: float, fmax: float, top_pad_hz: float,
                              ticks_khz: tuple[int, ...],
                              dpi: int, out_png: Path) -> None:
    """Pass 1b: render the static left-margin Hz-label overlay.

    Dimensions: yaxis_w × strip_h. The Hz tick labels (formatted as "<N>kHz")
    occupy the top (strip_h - xaxis_h) px so they align row-for-row with the
    spectrogram data in the wide strip; the bottom xaxis_h px are blank to
    leave room for the scrolling time-label strip's left edge.

    The axes y-range is [fmin, fmax + top_pad_hz] -- identical to the wide
    spec strip's y-range -- so tick positions line up exactly with the spec
    data rows, and the topmost tick (at fmax) sits a few px below the figure
    top edge so its label doesn't clip.
    """
    spec_h = strip_h - xaxis_h
    fig = plt.figure(figsize=(yaxis_w / dpi, strip_h / dpi), dpi=dpi,
                     facecolor="black")
    # Axes box sits flush against the right edge; labels occupy the leftmost
    # ~55% of the panel, ticks sit at the right boundary pointing inward so
    # they read as continuous with the spec data that begins one pixel further.
    ax = fig.add_axes((0.55, xaxis_h / strip_h, 0.45, spec_h / strip_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(fmin, fmax + top_pad_hz)
    ax.set_xticks([])
    ax.set_facecolor("black")
    ticks_hz = [t * 1000 for t in ticks_khz]
    ax.yaxis.tick_right()
    ax.set_yticks(ticks_hz)
    ax.set_yticklabels([f"{t}kHz" for t in ticks_khz],
                        color="white", fontsize=9)
    # tick marks on the right edge, labels on the left
    ax.tick_params(axis="y", which="both", colors="white", direction="in",
                   length=5, pad=2, left=False, labelleft=True, labelright=False,
                   right=True)
    for side in ("top", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["right"].set_color("white")
    fig.savefig(out_png, dpi=dpi, facecolor="black")
    plt.close(fig)


def build_main_filter(flips: list[str], video_w: int, video_h: int,
                      strip_h: int, segments: list[dict],
                      aud_idx: int, audio_fs: int, pitch: int,
                      drift_ratio: float, speed: float,
                      centerline: bool, duration: float) -> str:
    """Pass 2 graph: a tiled spectrogram row stacked over side-by-side videos.

    The top row is built from ``segments`` (left -> right), one per contiguous
    block of camera columns. Each segment is either:

      * a spectrogram strip -- the wide spec PNG (``seg['img_idx']``) cropped to
        ``data_w × strip_h`` with a left edge advancing at ``pps`` px/s, padded
        on the left by ``yaxis_w`` of black, with the static y-axis PNG
        (``seg['yax_idx']``) overlaid at x=0; or
      * a blank panel -- ``seg['width'] × strip_h`` of black for columns that
        have no spectrogram.

    All segments are forced to yuv420p so the ``hstack`` (and the later
    ``vstack`` with the videos) get matching pixel formats. Input order:
    videos [0..n-1], then (spec PNG, y-axis PNG) per spec segment, then audio.
    """
    n_videos = len(flips)
    parts = []
    for i, flip in enumerate(flips):
        chain = f"scale={video_w}:{video_h}"
        if flip:
            chain += f",{flip}"
        chain += f",fps={FPS},setpts=PTS-STARTPTS"
        parts.append(f"[{i}:v]{chain}[v{i}]")
    vlabels = "".join(f"[v{i}]" for i in range(n_videos))
    if n_videos > 1:
        parts.append(f"{vlabels}hstack=inputs={n_videos},format=yuv420p[videos]")
    else:
        parts.append(f"{vlabels}format=yuv420p[videos]")

    # Build each top-row segment, in left-to-right order.
    seg_labels = []
    for k, seg in enumerate(segments):
        lbl = f"seg{k}"
        if seg["kind"] == "blank":
            parts.append(
                f"color=c=black:s={seg['width']}x{strip_h}:r={FPS}:"
                f"d={duration:g},format=yuv420p[{lbl}]"
            )
            seg_labels.append(lbl)
            continue
        img_idx = seg["img_idx"]
        yax_idx = seg["yax_idx"]
        data_w = seg["data_w"]
        strip_w = seg["strip_w"]
        yaxis_w = seg["yaxis_w"]
        pps = seg["pps"]
        maxx = seg["maxx"]
        # Moving crop of the wide spec PNG: left edge = min(pps*t, maxx); \,
        # escapes the comma so min() is read as one crop option value.
        parts.append(
            f"[{img_idx}:v]crop={data_w}:{strip_h}:"
            f"x=min({pps:.6f}*t\\,{maxx}):y=0,"
            f"pad={strip_w}:{strip_h}:x={yaxis_w}:y=0:color=black,"
            f"setsar=1,setpts=PTS-STARTPTS[spad{k}]"
        )
        # Static y-axis overlay on the left margin; it's a still image, so
        # `overlay` consumes one frame and holds it (shortest=0, repeatlast=1
        # are overlay defaults but make them explicit).
        parts.append(
            f"[spad{k}][{yax_idx}:v]"
            f"overlay=x=0:y=0:shortest=0:repeatlast=1[slab{k}]"
        )
        if centerline:
            cx = yaxis_w + data_w // 2 - CENTERLINE_W // 2
            parts.append(
                f"[slab{k}]drawbox=x={cx}:y=0:w={CENTERLINE_W}:h={strip_h}:"
                f"color=white:t=fill,format=yuv420p[{lbl}]"
            )
        else:
            parts.append(f"[slab{k}]format=yuv420p[{lbl}]")
        seg_labels.append(lbl)

    toplabels = "".join(f"[{l}]" for l in seg_labels)
    if len(seg_labels) > 1:
        parts.append(f"{toplabels}hstack=inputs={len(seg_labels)}[spectop]")
    else:
        parts.append(f"{toplabels}null[spectop]")

    # Final video stream: stack the spec row on top of the camera grid, then
    # apply setpts=PTS/speed to retime to the requested playback speed. fps=FPS
    # resamples so the output stays at a standard framerate (drops frames for
    # speed>1, duplicates for speed<1) -- without it the output PTS would imply
    # a non-standard fps that some players handle awkwardly.
    parts.append(
        f"[spectop][videos]vstack=inputs=2,"
        f"setpts=PTS/{speed:.6f},fps={FPS}[outv]"
    )
    # Pitch shift DOWN by `pitch` via sample-rate manipulation -- no algorithmic
    # latency, unlike rubberband (which buffers ~100ms internally and would
    # delay the muxed audio so events arrive after the video frame that shows
    # them). asetrate reinterprets samples at (fs*drift_ratio)/pitch -- the
    # drift_ratio bakes the audio/video clock-drift correction into the same
    # rate-change step (so audio plays at its true real-world rate, matching
    # the video timeline), and dividing by pitch shifts the pitch down by
    # `pitch`x. aresample brings the rate back to standard 48 kHz; the atempo
    # chain speeds playback up by `pitch*speed` so duration matches the (sped-
    # up) video again, with pitch preserved at the pitch-shifted level. atempo
    # is split into <=2x steps because a single atempo=10 trips a known
    # assertion in ffmpeg's af_atempo.c.
    new_rate = int(round(audio_fs * drift_ratio / pitch))
    tempo_chain = _atempo_chain(float(pitch * speed))
    parts.append(
        f"[{aud_idx}:a]asetrate={new_rate},aresample=48000,{tempo_chain},"
        f"asetpts=PTS-STARTPTS[outa]"
    )
    return ";".join(parts)


def _atempo_chain(factor: float, max_step: float = 2.0) -> str:
    """Chain atempo filters so each step stays within [1/max_step, max_step].

    ffmpeg's af_atempo nominally accepts 0.5..100 in a single call, but extreme
    factors (e.g. 10) can trigger an internal assertion. Chaining keeps every
    step in the well-tested 0.5..2.0 range and multiplies to the same factor.
    """
    if 1.0 / max_step <= factor <= max_step:
        return f"atempo={factor:g}"
    parts, f = [], factor
    if factor > 1.0:
        while f > max_step:
            parts.append(f"atempo={max_step:g}")
            f /= max_step
    else:
        while f < 1.0 / max_step:
            parts.append(f"atempo={1.0 / max_step:g}")
            f *= max_step
    if abs(f - 1.0) > 1e-6:
        parts.append(f"atempo={f:g}")
    return ",".join(parts)


def parse_duration(value):
    """Normalise a duration to a positive number of seconds, or None for 'full'."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("full", "all", ""):
        return None
    seconds = float(value)
    return None if seconds <= 0 else seconds


def parse_specs_arg(tokens: list[str]) -> list[dict]:
    """Parse ``--specs`` tokens 'view[+view...]:channel' into strip dicts.

    Returns a list of ``{"channel": int, "views": [view, ...]}`` in the order
    given. A token with several '+'-joined views defines one strip stretched
    across those (contiguous) camera columns.
    """
    strips = []
    for tok in tokens:
        if ":" not in tok:
            raise ValueError(
                f"bad --specs token {tok!r}; expected 'view[+view...]:channel'")
        views_part, ch_part = tok.rsplit(":", 1)
        views = [v for v in views_part.split("+") if v]
        if not views:
            raise ValueError(f"bad --specs token {tok!r}; no views before ':'")
        try:
            channel = int(ch_part)
        except ValueError:
            raise ValueError(
                f"bad --specs token {tok!r}; channel {ch_part!r} is not an int")
        strips.append({"channel": channel, "views": views})
    return strips


def plan_top_row(views: list[str], strips: list[dict]) -> list[dict]:
    """Tile the camera columns into an ordered list of top-row segments.

    Each strip must cover a CONTIGUOUS left-to-right block of ``views`` and the
    strips may not overlap. Columns covered by no strip become ``blank``
    segments. Returns segments left -> right, each either
    ``{"kind": "spec", "strip": <index into strips>, "cols": n}`` or
    ``{"kind": "blank", "cols": n}``.
    """
    n = len(views)
    pos = {}
    for i, v in enumerate(views):
        pos.setdefault(v, i)
    col_strip: list[int | None] = [None] * n
    for si, strip in enumerate(strips):
        idxs = []
        for v in strip["views"]:
            if v not in pos:
                raise ValueError(
                    f"strip view {v!r} is not in --views {views}")
            idxs.append(pos[v])
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            raise ValueError(
                f"strip views {strip['views']} are not a contiguous "
                f"left-to-right block of --views {views}")
        for i in idxs:
            if col_strip[i] is not None:
                raise ValueError(
                    f"view {views[i]!r} is assigned to more than one strip")
            col_strip[i] = si
    segments = []
    i = 0
    while i < n:
        si = col_strip[i]
        j = i
        while j < n and col_strip[j] == si:
            j += 1
        if si is None:
            segments.append({"kind": "blank", "cols": j - i})
        else:
            segments.append({"kind": "spec", "strip": si, "cols": j - i})
        i = j
    return segments


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render a multi-camera video + centered-playhead "
                    "spectrogram mp4. Defaults come from the CONFIG block in "
                    "this file; flags below override them for one-off runs.")
    p.add_argument("-n", "--filenum", type=int, default=FILENUM,
                   help=f"file index (default {FILENUM})")
    p.add_argument("-s", "--start", type=float, default=START,
                   help=f"window start in seconds (default {START})")
    p.add_argument("-t", "--duration", default=DURATION,
                   help=f"window length in seconds, or 'full' (default {DURATION})")
    p.add_argument("--views", nargs="+", default=VIDEO_VIEWS,
                   help=f"camera views, left to right (default {VIDEO_VIEWS})")
    p.add_argument("--specs", nargs="+", default=None,
                   help="spectrogram strips as 'view[+view...]:channel' tokens; "
                        "a strip listing several views is stretched above them, "
                        "and views left uncovered get blank space (default: "
                        "SPEC_STRIPS config block)")
    p.add_argument("--audio-channel", type=int, default=AUDIO_CHANNEL,
                   help="mic channel for the muxed pitched audio (default: "
                        "first strip's channel)")
    p.add_argument("-d", "--datadir", default=DATADIR, help="input data directory")
    p.add_argument("-o", "--output", default=None,
                   help="output mp4 path (default: auto-named in OUTDIR)")
    p.add_argument("--av-offset", type=float, default=AV_SYNC_OFFSET_S,
                   help=f"extra constant seconds added to every audio read on"
                        f" TOP of automatic clock-drift correction (default "
                        f"{AV_SYNC_OFFSET_S}). Use only if a residual constant"
                        f" offset remains after drift correction.")
    p.add_argument("--no-drift", dest="drift", action="store_false",
                   help="disable automatic audio/video clock-drift correction"
                        " (drift_ratio = audio_dur / video_dur from ffprobe).")
    p.set_defaults(drift=True)
    p.add_argument("--speed", type=float, default=SPEED,
                   help=f"output playback speed; 1.0 = normal, 1.5 = 1.5x"
                        f" faster, 0.5 = half speed (default {SPEED}). Both"
                        f" video and audio are retimed; audio pitch is"
                        f" preserved at the sped-up rate.")
    args = p.parse_args()
    if args.speed <= 0:
        print(f"ERROR: --speed must be > 0 (got {args.speed})", file=sys.stderr)
        return 1

    # Spectrogram strips: from --specs if given, else the SPEC_STRIPS config.
    if args.specs:
        try:
            strips = parse_specs_arg(args.specs)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    else:
        strips = [{"channel": s["channel"], "views": list(s["views"])}
                  for s in SPEC_STRIPS]
    if not strips:
        print("ERROR: no spectrogram strips defined (SPEC_STRIPS / --specs)",
              file=sys.stderr)
        return 1
    try:
        seg_plan = plan_top_row(args.views, strips)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    datadir = Path(args.datadir)
    audio_channel = (args.audio_channel if args.audio_channel is not None
                     else strips[0]["channel"])
    try:
        videos = [_resolve_file(datadir, _video_candidates(v, args.filenum))
                  for v in args.views]
        strip_audios = [_resolve_file(datadir,
                                      _audio_candidates(s["channel"], args.filenum))
                        for s in strips]
        audio = _resolve_file(datadir,
                               _audio_candidates(audio_channel, args.filenum))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # geometry
    n = len(videos)
    video_h = _even(VIDEO_W * VIDEO_ASPECT)
    strip_h = _even(video_h * SPEC_FRAC)             # full spec-strip height
    total_w = VIDEO_W * n                            # even (VIDEO_W is even)
    yaxis_w = _even(SPEC_YAXIS_W)                    # static Hz-label panel
    xaxis_h = _even(SPEC_XAXIS_H)                    # scrolling time-label band
    flips = [flip_filter(VIDEO_FLIP.get(v, "")) for v in args.views]

    # window: resolve 'full', then expand by SPEC_WINDOW/2 on each side so the
    # spectrogram image has the past/future the centered crop needs.
    file_dur = ffprobe_duration(audio)
    audio_fs = sf.info(str(audio)).samplerate
    # Audio/video clock-drift correction: the WAV file and the MP4 file cover
    # the same real-world recording but with slightly mismatched sample clocks
    # (typically audio capture rate is a few Hz off from the nominal 125 kHz),
    # so their reported durations differ by ~0.07% on a ~360s file. The drift
    # ratio = audio_file_duration / video_file_duration scales every audio
    # time so it stays aligned to the video timeline; setting --no-drift skips
    # this if a future file pair turns out to be truly synced.
    video_dur = ffprobe_duration(videos[0])
    drift_ratio = file_dur / video_dur if args.drift else 1.0
    duration = parse_duration(args.duration)
    if duration is None:
        duration = file_dur - args.start
    half = SPEC_WINDOW / 2.0
    s0 = args.start - half
    s1 = args.start + duration + half
    span_len = s1 - s0  # = duration + SPEC_WINDOW

    # Per-strip geometry: each strip spans `cols` camera columns and so is
    # `cols * VIDEO_W` wide; the data area is that minus the Hz-label panel.
    for strip in strips:
        cols = len(strip["views"])
        strip_w = VIDEO_W * cols
        data_w = strip_w - yaxis_w
        if data_w <= 0:
            print(f"ERROR: SPEC_YAXIS_W ({yaxis_w}) >= strip width ({strip_w}) "
                  f"for views {strip['views']}", file=sys.stderr)
            return 2
        pps = data_w / SPEC_WINDOW
        full_w = int(round(pps * span_len))
        pps = full_w / span_len      # actual scale after integer rounding
        strip["geom"] = {"strip_w": strip_w, "data_w": data_w, "full_w": full_w,
                         "pps": pps, "maxx": full_w - data_w}

    if args.output:
        out = Path(args.output)
    else:
        views_tag = "+".join(args.views)
        chans_tag = "ch" + "-".join(str(s["channel"]) for s in strips)
        window = f"{args.start:g}+{duration:g}s"
        speed_tag = f"_{args.speed:g}x" if args.speed != 1.0 else ""
        out = Path(OUTDIR) / (
            f"sync_exp{EXP}_{chans_tag}_file{args.filenum}_{views_tag}"
            f"_{window}{speed_tag}.mp4")

    print(f">>> views : {', '.join(args.views)}  "
          f"({total_w}x{strip_h + video_h}, spec strip {strip_h}px, "
          f"yaxis {yaxis_w}px, xaxis {xaxis_h}px)")
    print(">>> specs :")
    for seg in seg_plan:
        if seg["kind"] == "blank":
            print(f"    [blank]  over {seg['cols']} column(s)")
        else:
            strip = strips[seg["strip"]]
            print(f"    ch{strip['channel']:<3} over {'+'.join(strip['views'])}")
    print(f">>> audio : {audio}  (ch{audio_channel}, pitch /{PITCH}, "
          f"drift ratio {drift_ratio:.6f}, av offset +{args.av_offset:.3f}s)")
    print(f">>> window: {args.start:g}s for {duration:g}s of source "
          f"(spec window {SPEC_WINDOW:g}s, 'now' centered, "
          f"playback speed {args.speed:g}x = {duration/args.speed:g}s of output)")
    print(f">>> output: {out}\n")

    with tempfile.TemporaryDirectory() as tmp:
        yaxis_png = Path(tmp) / "yaxis.png"

        # pass 1a: one wide scrolling spectrogram strip per spec, each from its
        # own mic channel but sharing the same [s0, s1] time window.
        for si, strip in enumerate(strips):
            g = strip["geom"]
            spec_png = Path(tmp) / f"spec_{si}.png"
            saud = strip_audios[si]
            sdur = ffprobe_duration(saud)
            render_spec_strip_png(saud, s0, s1, sdur,
                                  drift_ratio, args.av_offset,
                                  g["full_w"], strip_h, xaxis_h,
                                  SPEC_CMAP, None, SPEC_VMAX_DB, SPEC_DRANGE_DB,
                                  SPEC_FMIN, SPEC_FMAX, SPEC_TOP_PAD_HZ,
                                  SPEC_NFFT, SPEC_HOP, SPEC_WINDOW_FN,
                                  SPEC_DPI, spec_png)
            strip["png"] = spec_png
        # pass 1b: static y-axis (Hz) overlay -- identical for every strip, so
        # render once and reuse it as an input for each spec segment.
        render_yaxis_overlay_png(yaxis_w, strip_h, xaxis_h,
                                  SPEC_FMIN, SPEC_FMAX, SPEC_TOP_PAD_HZ,
                                  SPEC_YTICKS_KHZ, SPEC_DPI, yaxis_png)

        # pass 2: videos + moving crops of the spectrograms + pitched audio
        cmd = ["ffmpeg", "-y", "-hide_banner"]
        for v in videos:                                   # video inputs [0..n-1]
            cmd += ["-ss", f"{args.start:g}", "-t", f"{duration:g}", "-i", str(v)]
        # Add (spec PNG, y-axis PNG) inputs per spec segment, in left->right
        # order, and assign each segment its filter-graph input indices.
        idx = n
        segments = []
        for seg in seg_plan:
            if seg["kind"] == "blank":
                segments.append({"kind": "blank", "width": seg["cols"] * VIDEO_W})
                continue
            strip = strips[seg["strip"]]
            g = strip["geom"]
            cmd += ["-loop", "1", "-framerate", str(FPS),
                    "-t", f"{duration:g}", "-i", str(strip["png"])]
            img_idx = idx
            cmd += ["-loop", "1", "-framerate", str(FPS),
                    "-t", f"{duration:g}", "-i", str(yaxis_png)]
            yax_idx = idx + 1
            idx += 2
            segments.append({"kind": "spec", "img_idx": img_idx, "yax_idx": yax_idx,
                             "strip_w": g["strip_w"], "yaxis_w": yaxis_w,
                             "data_w": g["data_w"], "pps": g["pps"],
                             "maxx": g["maxx"]})
        # Audio extraction is scaled by drift_ratio: start*drift + offset
        # picks the audio-file moment that corresponds to video-time `start`,
        # and reading duration*drift of file content gives the right number
        # of samples for the corrected asetrate+atempo chain to play back in
        # exactly `duration` seconds (matching the video).
        audio_start = args.start * drift_ratio + args.av_offset
        audio_dur   = duration * drift_ratio
        aud_idx = idx
        cmd += ["-ss", f"{audio_start:.6f}",
                "-t", f"{audio_dur:.6f}", "-i", str(audio)]

        filtergraph = build_main_filter(flips, VIDEO_W, video_h, strip_h,
                                        segments, aud_idx,
                                        audio_fs, PITCH, drift_ratio,
                                        args.speed, SPEC_CENTERLINE, duration)
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
