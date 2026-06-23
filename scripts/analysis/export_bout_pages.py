#!/usr/bin/env python
"""Export bout spectrogram pages for talks.

Standalone script (runs as a fresh process so a misbehaving Jupyter kernel can't
crash on us). For each date, all bouts that pass the size + duration filters
are rendered, paginated 20 per sheet, in chronological order.

Per row:
    * Title bar:   exp/file · time-of-day · location · bout N · K calls · X.X s
    * Marker strip: a short black tick at each call's start; inter-call interval
                    (s, 2 decimals) labelled between consecutive ticks.
    * Spectrogram: 0-60 kHz, bright magma, x-ticks + labels every 1 s.

Event grouping:
    * Consecutive bouts on a page that belong to the same alarm event get a
      thick green vertical line down their shared left margin.

Output per date:
    BASE_PROCESSED_AUDIO/alarm/bout_example_pages/{date}/
        alarm_bout_examples_{date}_sheet01.png
        alarm_bout_examples_{date}_sheet01.pdf
        ... per sheet ...
        alarm_bout_examples_{date}.pdf   (multi-page, all sheets bundled)

Run with:
    python scripts/export_bout_pages.py
    python scripts/export_bout_pages.py --dates 2025_03 2025_07
    # above-ground only:
    python scripts/export_bout_pages.py \
        --locations arena_1 arena_2 \
        --out-dir BASE_PROCESSED_AUDIO/alarm/bout_example_pages_above_ground
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from vocalization_analysis.acoustic_features import load_call_slice
from vocalization_analysis.bouts import detect_alarm_scales, summarize_scale

# --------------------------------------------------------------------------
# Paths (cluster only - need raw WAVs for the audio loading).
# --------------------------------------------------------------------------
HOST = platform.system()
if HOST != "Linux":
    raise RuntimeError(
        "export_bout_pages.py must run on the Linux cluster (raw WAVs)."
    )

PARQUET_DIR = Path(
    "/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
    "Processed_data/Audio/all_calls/parquet_cache"
)
BASE_PROCESSED_AUDIO = Path(
    "/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio"
)
DEFAULT_OUT_DIR = BASE_PROCESSED_AUDIO / "alarm" / "bout_example_pages"

# --------------------------------------------------------------------------
# Config (tweakable via CLI; the heavier knobs live here).
# --------------------------------------------------------------------------
DATES_TO_PLOT   = ["2025_03", "2025_07", "2025_10", "2026_02"]
N_PER_PAGE      = 20
MIN_BOUT_SIZE   = 5
MAX_BOUT_DUR_S  = 20.0
PRE_SEC         = 0.1
POST_SEC        = 0.3
NFFT            = 512
HOP             = 256
DPI             = 110
CMAP            = "magma"
VMIN, VMAX      = -65, -10           # brighter dynamic range
ROW_HEIGHT_IN   = 1.9              # taller spectrograms
PAGE_WIDTH_IN   = 28
SPEC_FMAX_KHZ   = 60               # spectrogram y-axis upper limit
EVENT_LINE_COLOR  = "#1f9d55"
EVENT_LINE_WIDTH  = 3.5
EVENT_LINE_INSET  = 0.020          # figure-relative offset left of axes (clears "kHz" label)
EVENT_FOOT_LEN    = 0.010          # length of the [ bracket's horizontal feet


# --------------------------------------------------------------------------
# Data loading.
# --------------------------------------------------------------------------
def load_alarm_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    parts = []
    for date in DATES_TO_PLOT:
        df = pd.read_parquet(PARQUET_DIR / f"all_calls_{date}.parquet")
        df = df[df["event_type"] == "alarm"]
        parts.append(df)
    calls = pd.concat(parts, ignore_index=True)
    calls = detect_alarm_scales(calls)
    bouts_meta = summarize_scale(calls, prefix="bout")
    # bout_id -> event_id, for the green event-connector lines.
    bout_to_event = (
        calls.drop_duplicates("bout_id")
             .set_index("bout_id")["event_id"]
             .to_dict()
    )
    return calls, bouts_meta, bout_to_event


# --------------------------------------------------------------------------
# Page rendering.
# --------------------------------------------------------------------------
def _format_time_of_day(ts) -> str:
    try:
        return ts.strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001
        return str(ts)[-8:]


def _draw_bout_row(
    ax_mark: plt.Axes,
    ax_spec: plt.Axes,
    *,
    bout_calls: pd.DataFrame,
    bout_row: pd.Series,
    x_lim: tuple[float, float],
) -> bool:
    """Render one bout into the two stacked axes. Returns True on success."""
    if bout_calls.empty:
        return False
    if bout_calls["file_num"].nunique() != 1 or bout_calls["channel"].nunique() != 1:
        return False

    first_call = bout_calls.iloc[0]
    last_call  = bout_calls.iloc[-1]
    bout_start_abs = first_call["start_time_file_sec"]
    win_start = max(0.0, bout_start_abs - PRE_SEC)
    win_stop  = last_call["stop_time_file_sec"] + POST_SEC

    try:
        y, sr = load_call_slice(
            BASE_PROCESSED_AUDIO,
            first_call["date_folder"], first_call["exp"],
            first_call["channel"],     first_call["file_num"],
            win_start, win_stop, pad_sec=0.0,
        )
    except Exception:  # noqa: BLE001
        return False
    if len(y) < NFFT:
        return False

    actual_dur      = len(y) / sr
    win_stop_actual = win_start + actual_dur

    S    = np.abs(librosa.stft(y.astype(np.float32),
                                n_fft=NFFT, hop_length=HOP, window="hann"))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    rel_start = win_start - bout_start_abs
    rel_stop  = win_stop_actual - bout_start_abs
    extent = [rel_start, rel_stop, freqs[0] / 1000, freqs[-1] / 1000]

    # Spectrogram (bright).
    ax_spec.imshow(
        S_db, aspect="auto", origin="lower",
        cmap=CMAP, vmin=VMIN, vmax=VMAX, extent=extent,
    )
    ax_spec.set_ylim(0, SPEC_FMAX_KHZ)
    ax_spec.set_xlim(*x_lim)
    ax_spec.set_ylabel("kHz", fontsize=8)
    # X-ticks + labels every 1 s on every row.
    xticks = np.arange(0, MAX_BOUT_DUR_S + 0.001, 1.0)
    ax_spec.set_xticks(xticks)
    ax_spec.tick_params(labelsize=7)

    # Marker strip. NB the x-axis is shared with ax_spec, so we must NOT call
    # ax_mark.set_xticks([]) (that would wipe out the spec's ticks too). Instead
    # we hide ax_mark's own tick marks/labels via tick_params.
    ax_mark.set_xlim(*x_lim)
    ax_mark.set_ylim(0, 1)
    ax_mark.set_yticks([])
    ax_mark.tick_params(
        axis="x", bottom=False, top=False,
        labelbottom=False, labeltop=False,
    )
    for spine in ax_mark.spines.values():
        spine.set_visible(False)

    prev_stop_rel = None
    for j, cr in enumerate(bout_calls.itertuples(index=False)):
        t_rel = cr.start_time_file_sec - bout_start_abs
        # Short vertical tick at this call's START.
        ax_mark.plot([t_rel, t_rel], [0.0, 0.35],
                     "-", color="black", linewidth=1.5)
        # ICI label (seconds) BETWEEN consecutive calls.
        if j > 0 and prev_stop_rel is not None:
            ici_s = cr.ici_s if not pd.isna(cr.ici_s) else None
            if ici_s is not None:
                # Position at the midpoint of the gap between prev_stop and t_rel.
                mid = (prev_stop_rel + t_rel) / 2
                ax_mark.text(
                    mid, 0.40, f"{ici_s:.2f}s",
                    ha="center", va="bottom",
                    fontsize=6, color="black",
                )
        prev_stop_rel = cr.stop_time_file_sec - bout_start_abs

    tod_str = _format_time_of_day(bout_row["start_time"])
    loc_str = str(first_call.get("assigned_location", "?"))
    ax_mark.set_title(
        f"exp{int(first_call['exp'])}/f{int(first_call['file_num']):03d}  ·  "
        f"{tod_str}  ·  {loc_str}  ·  bout {int(bout_row['bout_id'])}  ·  "
        f"{int(bout_row['bout_size'])} calls  ·  {bout_row['duration_s']:.1f}s",
        fontsize=9, loc="left", pad=12,
    )
    return True


def render_sheet(
    *,
    date: str,
    sheet_idx: int,
    n_sheets: int,
    chunk: pd.DataFrame,
    calls: pd.DataFrame,
    bout_to_event: dict[int, int],
) -> plt.Figure:
    """Render one sheet (up to N_PER_PAGE bouts) and return the Figure."""
    plt.ioff()
    fig = plt.figure(
        figsize=(PAGE_WIDTH_IN, ROW_HEIGHT_IN * N_PER_PAGE),
        facecolor="white",
    )
    outer = fig.add_gridspec(
        N_PER_PAGE, 1,
        hspace=0.55,
        top=0.985, bottom=0.015, left=0.04, right=0.99,
    )
    x_lim = (-PRE_SEC, MAX_BOUT_DUR_S + POST_SEC)

    # Track per-row axes + event_ids for the green-line pass below.
    mark_axes: list[plt.Axes | None] = [None] * N_PER_PAGE
    spec_axes: list[plt.Axes | None] = [None] * N_PER_PAGE
    event_ids: list[int | None] = [None] * N_PER_PAGE

    for slot in range(N_PER_PAGE):
        if slot >= len(chunk):
            break
        bout_row   = chunk.iloc[slot]
        bout_id    = int(bout_row["bout_id"])
        bout_calls = calls[calls["bout_id"] == bout_id].sort_values(
            "start_time_file_sec"
        )

        inner = outer[slot].subgridspec(
            2, 1, height_ratios=[0.22, 1.7], hspace=0.05,
        )
        ax_mark = fig.add_subplot(inner[0])
        ax_spec = fig.add_subplot(inner[1], sharex=ax_mark)
        mark_axes[slot] = ax_mark
        spec_axes[slot] = ax_spec
        event_ids[slot] = bout_to_event.get(bout_id)

        ok = _draw_bout_row(
            ax_mark, ax_spec,
            bout_calls=bout_calls, bout_row=bout_row, x_lim=x_lim,
        )
        if not ok:
            ax_mark.set_axis_off()
            ax_spec.set_axis_off()
            mark_axes[slot] = None
            spec_axes[slot] = None

    # Event-connector brackets (figure-coordinate space).
    # Identify maximal runs of consecutive rows that share an event_id; for
    # each run of length >= 2 we draw a `[`-shaped bracket to the left of the
    # row's y-axis (kHz label). The bracket has a vertical line spanning the
    # group's rows plus short horizontal "feet" at the top and bottom.
    fig.canvas.draw()

    # Group consecutive same-event rows.
    groups: list[tuple[int, int, int]] = []
    g_start: int | None = None
    g_event: int | None = None
    for r in range(N_PER_PAGE):
        eid = event_ids[r] if mark_axes[r] is not None else None
        if eid is None:
            if g_start is not None and g_event is not None:
                groups.append((g_start, r - 1, g_event))
            g_start, g_event = None, None
            continue
        if g_event is None:
            g_start, g_event = r, eid
        elif eid != g_event:
            groups.append((g_start, r - 1, g_event))
            g_start, g_event = r, eid
    if g_start is not None and g_event is not None:
        groups.append((g_start, N_PER_PAGE - 1 if mark_axes[N_PER_PAGE - 1] is not None
                        else (g_start), g_event))
    # Keep only runs of >= 2 rows.
    groups = [(s, e, _) for (s, e, _) in groups if e > s]

    for g_top_row, g_bot_row, _eid in groups:
        ax_top = mark_axes[g_top_row]
        ax_bot = spec_axes[g_bot_row]
        if ax_top is None or ax_bot is None:
            continue
        bb_top = ax_top.get_position()
        bb_bot = ax_bot.get_position()
        x_vert = bb_top.x0 - EVENT_LINE_INSET
        y_top  = bb_top.y1
        y_bot  = bb_bot.y0
        # Vertical spine of the bracket.
        fig.add_artist(Line2D(
            [x_vert, x_vert], [y_bot, y_top],
            transform=fig.transFigure,
            color=EVENT_LINE_COLOR, linewidth=EVENT_LINE_WIDTH,
            alpha=0.85, solid_capstyle="butt",
        ))
        # Top foot (extends right from the spine).
        fig.add_artist(Line2D(
            [x_vert, x_vert + EVENT_FOOT_LEN], [y_top, y_top],
            transform=fig.transFigure,
            color=EVENT_LINE_COLOR, linewidth=EVENT_LINE_WIDTH,
            alpha=0.85, solid_capstyle="butt",
        ))
        # Bottom foot.
        fig.add_artist(Line2D(
            [x_vert, x_vert + EVENT_FOOT_LEN], [y_bot, y_bot],
            transform=fig.transFigure,
            color=EVENT_LINE_COLOR, linewidth=EVENT_LINE_WIDTH,
            alpha=0.85, solid_capstyle="butt",
        ))

    fig.suptitle(
        f"{date}  ·  Sheet {sheet_idx + 1}/{n_sheets}  (chronological, all bouts)",
        y=0.996, fontsize=14, fontweight="bold",
    )
    return fig


def chunk_bouts(pool: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a date's bouts into chronologically ordered pages of N_PER_PAGE."""
    pool = pool.sort_values("start_time").reset_index()
    n_sheets = math.ceil(len(pool) / N_PER_PAGE)
    return [
        pool.iloc[k * N_PER_PAGE : (k + 1) * N_PER_PAGE]
        for k in range(n_sheets)
    ]


def render_one_date(
    *,
    date: str,
    pool: pd.DataFrame,
    calls: pd.DataFrame,
    bout_to_event: dict[int, int],
    out_root: Path,
) -> None:
    date_dir = out_root / date
    date_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_bouts(pool)
    n_sheets = len(chunks)
    print(f"  {date}: {len(pool)} bouts → {n_sheets} sheets, dir: {date_dir}")

    pdf_path = date_dir / f"alarm_bout_examples_{date}.pdf"
    with PdfPages(pdf_path) as pdf:
        for sheet_idx, chunk in enumerate(chunks):
            if chunk.empty:
                continue
            fig = render_sheet(
                date=date,
                sheet_idx=sheet_idx,
                n_sheets=n_sheets,
                chunk=chunk,
                calls=calls,
                bout_to_event=bout_to_event,
            )
            # PNG sheet.
            png_path = date_dir / (
                f"alarm_bout_examples_{date}_sheet{sheet_idx + 1:02d}.png"
            )
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight",
                        facecolor="white")
            # Multi-page PDF and per-sheet PDF (the per-sheet one is convenient
            # for picking individual slides without scrolling through the bundle).
            sheet_pdf_path = date_dir / (
                f"alarm_bout_examples_{date}_sheet{sheet_idx + 1:02d}.pdf"
            )
            fig.savefig(sheet_pdf_path, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            plt.close("all")
            gc.collect()
            print(f"    sheet {sheet_idx + 1}/{n_sheets} → {png_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--dates", nargs="+", default=DATES_TO_PLOT,
        help="Subset of dates to render (default: all four)",
    )
    parser.add_argument(
        "--locations", nargs="+", default=None,
        help=(
            "Filter to bouts in these assigned_location values "
            "(e.g. --locations arena_1 arena_2 for above-ground). "
            "Default: all locations."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.out_dir}")
    print("Loading data…")
    calls, bouts_meta, bout_to_event = load_alarm_data()
    print(f"  {len(calls):,} alarm calls, {len(bouts_meta):,} bouts")

    if args.locations is not None:
        print(f"Location filter: {args.locations}")

    for date in args.dates:
        pool = bouts_meta[
            (bouts_meta["bout_size"]    >= MIN_BOUT_SIZE)
            & (bouts_meta["duration_s"] <= MAX_BOUT_DUR_S)
            & (bouts_meta["date_folder"] == date)
        ]
        if args.locations is not None:
            pool = pool[pool["assigned_location"].isin(args.locations)]
        if len(pool) == 0:
            print(f"  {date}: no bouts match filters - skipping")
            continue
        render_one_date(
            date=date,
            pool=pool,
            calls=calls,
            bout_to_event=bout_to_event,
            out_root=args.out_dir,
        )

    print(f"\nDone. PNGs + PDFs in: {args.out_dir}")


if __name__ == "__main__":
    main()