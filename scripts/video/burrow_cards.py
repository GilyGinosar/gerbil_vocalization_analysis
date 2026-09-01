#!/usr/bin/env python
"""Curation cards for burrow traverses, straight to JPG contact sheets.

A card is one traverse: the tunnel-mic spectrogram on top, DAS's detections as a
coloured ribbon under it, then the cached frame strip and a time axis. Reading
them side by side is how the detector gets checked against the audio -- a sweep
with no tick is a miss, a tick with no sweep is a false positive, and a tick in
the wrong compartment is a location-assignment error. All three would propagate
silently into every rate we compute.

Sheets, not a web page: the cards are big, and a browser is the wrong tool on a
remote shell. Each JPG opens straight in the VS Code editor over SSH.

Selection is either a spread across the whole date folder -- one traverse per
experiment, cycling, so a sample is not just whichever experiment is biggest --
or a targeted query with --position-band, which keeps only traverses whose calls
happened while the animal was in a given stretch of tunnel.

"With calls" means a tunnel-localised call in the card window, using each
experiment's own threshold, not merely any call in earshot -- that mostly tracks
whether the colony happened to be noisy.

    python scripts/video/burrow_cards.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --per-cell 15 --out-dir exports/sample_2026_02
    python scripts/video/burrow_cards.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --direction to_nest --position-band 0.05,0.15 --per-cell 60 \
        --out-dir exports/band_nestmouth

Writes sheets/<direction>_NN.jpg and selection.csv (which traverses got drawn).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pipeline.paths import BASE_RAW, experiment_audio_dir  # noqa: E402
from scripts.utils.data_rules import load_traverses  # noqa: E402
from scripts.video.burrow_scan import AFTER_S, BEFORE_S, TILE_FPS, TILE_W  # noqa: E402
from scripts.video.burrow_transit_picker import CALL_COLORS, t_to_x  # noqa: E402
from scripts.video.burrow_transit_picker import (  # noqa: E402
    annotate_spectrogram, audio_path, load_calls, mark_crossing, read_window,
    spectrogram_tile, time_axis, video_duration,
)

PX_PER_S = TILE_W * TILE_FPS
GUIDE_KHZ = 15
RIBBON_H = 13
LABEL_H = 30        # px of caption drawn above each card
LEGEND_H = 34       # px of colour key at the top of every sheet
SEPARATOR_H = 8     # px of dark gap between cards
USV = ("high-freq", "warble")
FPS = 30


# ---- what goes on a card --------------------------------------------------

def all_calls(exp: int) -> dict[str, dict[int, list]]:
    """Every DAS call, per compartment, with its type -- not just the USV subset.

    The ribbon is a check on the detector, so it must show what the detector
    actually said, including the types the analysis then discards.
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


CHANNEL_NAME = {1: "TUNNEL mic  (ch01)", 0: "NEST mic  (ch00, deeper in)",
                10: "ARENA_1 mic  (averaged ch10 — what DAS scored)"}
CHANNEL_LABEL_SCALE = 0.85      # the three traces are the point of the card; name
                                # them big enough to read on a slide
AVERAGED = (10, 20, 30)         # DAS ran on these, not on any single raw mic
NEST_PANEL_MAX_W = 1400   # only a sanity cap; the nest frame keeps its own aspect


def nest_frame(datadir: Path, file_num: int, t_entry: float,
               height: int, mirror: bool = False) -> np.ndarray | None:
    """One frame of the nest_top camera at the moment the animal enters the tunnel.

    The burrow_side camera cannot see who is already in the nest -- that is the
    whole reason the localiser exists -- so this puts the nest itself on the card.
    Same sync folder and the same time base as t_entry; the cameras are recorded
    together, but they are separate devices and nobody has verified frame-level
    alignment between them, so read it as "about then", not exactly then.
    """
    path = datadir / f"video_nest_top_{file_num:03d}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t_entry * fps), 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mirror:
        frame = cv2.flip(frame, 1)          # match the mirrored tunnel strip
    # keep the camera's own aspect: it is a wide view of the whole nest, and
    # cropping it to a card-sized square threw away most of the floor
    scale = min(height / frame.shape[0], NEST_PANEL_MAX_W / frame.shape[1])
    frame = cv2.resize(frame, (max(int(frame.shape[1] * scale), 1),
                               max(int(frame.shape[0] * scale), 1)))
    if frame.shape[0] < height:                       # letterbox to the card height
        pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), np.uint8)
        frame = cv2.vconcat([frame, pad])
    cv2.putText(frame, f"nest_top @ ENTRY  t={t_entry:.2f}s", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 235, 255), 1, cv2.LINE_AA)
    return frame


def arena_frame(datadir: Path, file_num: int, t_entry: float,
                height: int) -> np.ndarray | None:
    """One frame of the arena_1 camera at the moment the animal enters the tunnel.

    That instant is the useful one: it shows who was left OUT in the arena and so
    could not be the animal in the tunnel. arena_1 is `video_center` -- confirmed
    by matching per-video detection counts against files_vetted, where
    video_center rows equal the arena_1 totals exactly and video_gily_center
    equals arena_2.
    """
    path = datadir / f"video_center_{file_num:03d}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t_entry * fps), 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    scale = min(height / frame.shape[0], NEST_PANEL_MAX_W / frame.shape[1])
    frame = cv2.resize(frame, (max(int(frame.shape[1] * scale), 1),
                               max(int(frame.shape[0] * scale), 1)))
    if frame.shape[0] < height:
        pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), np.uint8)
        frame = cv2.vconcat([frame, pad])
    cv2.putText(frame, f"ARENA_1 @ ENTRY  t={t_entry:.2f}s", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 235, 255), 2, cv2.LINE_AA)
    return frame


ORIGIN_COLOUR = {"tunnel": (60, 190, 255), "nest": (235, 150, 70)}   # BGR


def localiser_ribbon(tunnel_starts, nest_starts, t0: float, t1: float,
                     width: int, reverse: bool) -> np.ndarray:
    """One row marking the localiser's verdict on every call in view.

    Amber = scored tunnel-origin (ch01 louder than that experiment's tunnel-empty
    reference), blue = nest-origin. Only the onset is known here, so each call is
    a fixed-width tick rather than a bar. Read it against the two spectrograms
    above: the verdict is a threshold on their level difference, and at the
    default cut fewer than half the amber ticks had an animal in the tunnel.
    """
    bar = np.full((RIBBON_H, width, 3), 24, np.uint8)
    counts = {"tunnel": 0, "nest": 0}
    for kind, starts in (("nest", nest_starts), ("tunnel", tunnel_starts)):
        for start in starts:
            if start < t0 or start > t1:
                continue
            counts[kind] += 1
            x = t_to_x(start, t0, t1, width, reverse)
            cv2.rectangle(bar, (max(0, x - 2), 2), (min(width, x + 2), RIBBON_H - 3),
                          ORIGIN_COLOUR[kind], -1)
    cv2.putText(bar, f"localiser  tunnel({counts['tunnel']})  nest({counts['nest']})",
                (4, RIBBON_H - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1)
    return bar


def build_card(scan: Path, row, direction: str, channels: tuple[int, ...],
               das_cache: dict, drift_cache: dict,
               with_nest_frame: bool = False,
               origin: tuple[dict, dict] | None = None,
               with_arena_frame: bool = False) -> np.ndarray | None:
    """Spectrograms (one per channel) + DAS ribbons + frame strip + time axis.

    With both ch01 and ch00 the localiser's own evidence is on the card: a call
    louder on the top strip than the bottom one is what it scores as
    tunnel-origin, and you can see whether that matches what you hear.
    """
    strip = cv2.imread(str(scan / str(row.exp) / row.tile))
    if strip is None:
        return None
    datadir = BASE_RAW / f"experiment_{row.exp}" / "concatenated_data_cam_mic_sync"
    key = (row.exp, row.file_num)
    if key not in drift_cache:
        # audio and video run on different clocks in these folders (~0.07%), so the
        # spectrogram must be stretched to the video's time base, not merely offset
        wav = audio_path(datadir, channels[0], row.file_num)
        cap = cv2.VideoCapture(str(datadir / row.video))
        with sf.SoundFile(str(wav)) as handle:
            audio_dur = handle.frames / handle.samplerate
        drift_cache[key] = audio_dur / max(video_duration(cap), 1e-9)
        cap.release()

    entry, out = float(row.t_entry), float(row.t_out)
    # A to_nest animal travels right-to-left in the crop, and we want travel to
    # read left-to-right on every sheet. Reversing the TIME axis achieves that but
    # mirrors the spectrogram with it, so every call sweep is drawn backwards --
    # a rising USV looks like a falling one, which is unreadable for call shape.
    # Mirror each tile's PIXELS instead and keep time running forward: the animal
    # still travels left-to-right, and the audio is drawn the way it sounded.
    reverse = False
    mirror = direction == "to_nest"
    if mirror:
        n_tiles = strip.shape[1] // TILE_W
        strip = cv2.hconcat([cv2.flip(strip[:, i * TILE_W:(i + 1) * TILE_W], 1)
                             for i in range(n_tiles)])
    width = strip.shape[1]
    t0 = entry - BEFORE_S
    t1 = t0 + width / PX_PER_S

    specs = []
    for ch in channels:
        if ch in AVERAGED:
            # channel_10/20/30 are the per-compartment averages DAS was run on;
            # they live under the processed Audio tree, not in the sync folder
            wav = (experiment_audio_dir(int(row.exp)) / "Averaged_wavs_w_annotations"
                   / f"channel_{ch}_file_{int(row.file_num):03d}.wav")
        else:
            wav = audio_path(datadir, ch, row.file_num)
        if wav is None or not Path(wav).exists():
            continue
        audio, fs = read_window(wav, t0, t1, drift_cache[key], 0.0)
        spec = spectrogram_tile(audio, fs, width, reverse)
        annotate_spectrogram(spec, t0, t1, entry, out, reverse)
        mark_crossing(spec, t0, t1, entry, out, BEFORE_S,
                      ("enters", "out of tunnel"), reverse)
        y = int(spec.shape[0] * (1 - (GUIDE_KHZ * 1000 - 500) / (45000 - 500)))
        cv2.line(spec, (0, y), (width, y), (110, 110, 110), 1)
        cv2.putText(spec, CHANNEL_NAME.get(ch, f"ch{ch:02d}"), (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, CHANNEL_LABEL_SCALE,
                    (120, 235, 255), 2, cv2.LINE_AA)
        specs.append(spec)
    axis = time_axis(width, t0, t1, entry, reverse)
    if row.exp not in das_cache:
        das_cache[row.exp] = all_calls(int(row.exp))
    under = das_cache[row.exp]["underground"].get(row.file_num, [])
    arena = das_cache[row.exp]["arena_1"].get(row.file_num, [])
    bars = [ribbon(under, t0, t1, width, reverse, "DAS underground"),
            ribbon(arena, t0, t1, width, reverse, "DAS arena_1")]
    if origin is not None:
        tunnel_starts, nest_starts = origin
        key = (int(row.exp), int(row.file_num))
        bars.insert(0, localiser_ribbon(tunnel_starts.get(key, ()),
                                        nest_starts.get(key, ()),
                                        t0, t1, width, reverse))
    card = cv2.vconcat([*specs, *bars, strip, axis])
    if with_nest_frame:
        # the panel goes on whichever edge the nest end is: to_nest cards mirror
        # their tiles so the nest sits on the RIGHT, to_arena cards do not. Putting
        # it beside the nest keeps the two views of the same place adjacent, and on
        # to_nest it also leaves t=0 at x=0 on every card, which the left-hand
        # placement did not -- the panel's width varies with the camera's aspect.
        face = nest_frame(datadir, int(row.file_num), entry, card.shape[0], mirror)
        if face is None:
            face = np.zeros((card.shape[0], 320, 3), np.uint8)
        card = cv2.hconcat([card, face] if mirror else [face, card])
    if with_arena_frame:
        # always on the LEFT: the arena is where the animal came FROM, so it reads
        # left-to-right as arena -> tunnel -> nest, matching the spectrogram order
        out_face = arena_frame(datadir, int(row.file_num), entry, card.shape[0])
        if out_face is None:
            out_face = np.zeros((card.shape[0], 320, 3), np.uint8)
        card = cv2.hconcat([out_face, card])
    return card


# ---- stacking cards into sheets -------------------------------------------

def legend(width: int) -> np.ndarray:
    """The call-type colour key, on every sheet.

    A JPG has no top bar to carry it, and without it the ribbon ticks are just
    unreadable colours.
    """
    bar = np.full((LEGEND_H, width, 3), 45, np.uint8)
    x = 10
    cv2.putText(bar, "call ribbon:", (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (215, 215, 215), 1, cv2.LINE_AA)
    x = 140
    for name, colour in CALL_COLORS.items():
        if name == "noise":
            continue
        cv2.rectangle(bar, (x, 10), (x + 16, 24), colour, -1)
        cv2.putText(bar, name, (x + 22, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (215, 215, 215), 1, cv2.LINE_AA)
        x += 40 + 12 * len(name)
    cv2.putText(bar, "green lines + shading = the traverse   |   scale bar top right = 1 s   |"
                     "   tunnel mic ch01, 0.5-45 kHz", (x + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 235, 150), 1, cv2.LINE_AA)
    return bar


def caption(width: int, text: str, calls: int) -> np.ndarray:
    """A dark caption bar; tinted when the card has calls in view."""
    bar = np.full((LABEL_H, width, 3), 34 if calls else 22, np.uint8)
    colour = (120, 235, 255) if calls else (170, 170, 170)
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    return bar


def build_sheets(cards: list[dict], out_dir: Path, per_sheet: int, quality: int,
                 align: str, sort_by: str, max_width: int, split_calls: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    groups: dict[str, list[dict]] = {}
    for card in cards:
        # vocal and quiet traverses on separate sheets: the question is whether calling
        # accompanies the passage, and mixing them makes that hard to see at a glance
        key = card["direction"] + ("_calls" if card["calls"] else "_quiet") if split_calls \
            else card["direction"]
        groups.setdefault(key, []).append(card)

    for name in sorted(groups):
        # Sheet width is set by its widest card, so grouping similar widths keeps
        # short traverses off a page stretched by one long one.
        if sort_by == "width":
            group = sorted(groups[name], key=lambda c: c["image"].shape[1])
        else:
            group = sorted(groups[name], key=lambda c: -c["calls"])
        # Pack cards into sheets: a page closes when it is full OR when adding the
        # next (wider) card would blow the width budget. Sheet width is set by its
        # widest member, so without this one 28 s traverse makes an 18000 px page
        # on which every other card is mostly padding.
        pages, page = [], []
        side = "left" if align == "auto" else align
        for card in group:
            wide = max([card["image"].shape[1]] + [c["image"].shape[1] for c in page])
            if page and (len(page) >= per_sheet or wide > max_width):
                pages.append(page)
                page = []
            page.append(card)
        if page:
            pages.append(page)

        position = 0
        for page_no, chunk in enumerate(pages, start=1):
            start = position
            position += len(chunk)
            width = max(card["image"].shape[1] for card in chunk)
            path = out_dir / f"{name}_{page_no:02d}.jpg"
            pieces = [legend(width)]
            for position, card in enumerate(chunk, start=start + 1):
                image = card["image"]
                if image.shape[1] != width:      # pad, never rescale: rescaling would
                    pad = np.zeros((image.shape[0], width - image.shape[1], 3), np.uint8)
                    # Pad on the side away from t=0 so the anchor column stays put.
                    # Left-to-right cards start at t0, so t=0 is a fixed offset from
                    # the LEFT and they pad right; right-to-left cards are the mirror.
                    image = (cv2.hconcat([pad, image]) if side == "right"
                             else cv2.hconcat([image, pad]))   # change the card's time scale
                pieces.append(caption(width, f"[{position}/{len(group)}] {card['label']}",
                                      card["calls"]))
                pieces.append(image)
                pieces.append(np.full((SEPARATOR_H, width, 3), 60, np.uint8))
                card["sheet"] = path.name
            sheet = cv2.vconcat(pieces)
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written.append(path)
            print(f"{path}  ({len(chunk)} crossings, {sheet.shape[1]}x{sheet.shape[0]})")
    return written


# ---- which traverses to draw ----------------------------------------------

def select_from_csv(traverses: pd.DataFrame, path: Path, tol: float = 0.01):
    """Keep only the traverses listed in a CSV of exp,file_num,t_entry.

    Written for the nest-motion work: the motion pass measures a sample of
    traverses and scores each one, and the question "show me the ones where the
    nest was still" needs those exact rows back, not a re-derivation. Matching is
    nearest-t_entry within `tol` rather than equality, because t_entry survives a
    round trip through CSV as text and the last digit does not always come back.

    Any extra column in the CSV rides along as `sel_<name>`, so a score computed
    there can be printed on the card.
    """
    want = pd.read_csv(path)
    extra = [c for c in want.columns if c not in ("exp", "file_num", "t_entry")]
    by_file: dict[tuple[int, int], list] = {}
    for r in want.itertuples():
        by_file.setdefault((int(r.exp), int(r.file_num)), []).append(r)
    keep, carried = [], {c: [] for c in extra}
    for row in traverses.itertuples():
        best, gap = None, tol
        for cand in by_file.get((int(row.exp), int(row.file_num)), ()):
            d = abs(float(cand.t_entry) - float(row.t_entry))
            if d <= gap:
                best, gap = cand, d
        keep.append(best is not None)
        for c in extra:
            carried[c].append(getattr(best, c) if best is not None else np.nan)
    out = traverses[keep].copy()
    for c in extra:
        out[f"sel_{c}"] = [v for v, k in zip(carried[c], keep) if k]
    print(f"--select-csv: matched {len(out)} of {len(want)} listed traverses")
    return out


def tunnel_localised(scan: Path) -> dict[tuple[int, int], np.ndarray]:
    """Calls the localiser puts in the tunnel, per experiment's own threshold."""
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
    return localised


def count_in_band(scan: Path, traverses: pd.DataFrame, lo: float, hi: float) -> list[int]:
    """How many of each traverse's calls happened inside a stretch of tunnel."""
    counts, cache = [], {}
    # one video holds several traverses and they arrive together, so remembering
    # just the last track read avoids re-reading the same parquet a few times over
    last_track: tuple[Path | None, tuple] = (None, ())
    for row in traverses.itertuples():
        if row.exp not in cache:
            calls = load_calls(int(row.exp))
            cache[row.exp] = {k: np.sort(np.array([c for c, _, t in v if t in USV]))
                              for k, v in calls.items()}
        times = cache[row.exp].get(row.file_num)
        n = 0
        if times is not None and times.size:
            track_path = scan / str(row.exp) / "tracks" / f"{Path(row.video).stem}.parquet"
            if track_path.exists():
                if last_track[0] != track_path:
                    track = pd.read_parquet(track_path)
                    last_track = (track_path,
                                  (track.x.to_numpy(), track.n_animals.to_numpy()))
                xs, na = last_track[1]
                a, b = int(row.t_entry * FPS), int(row.t_out * FPS)
                sel = times[(times >= row.t_entry) & (times <= row.t_out)]
                if sel.size and b < len(xs):
                    idx = np.clip((sel * FPS).astype(int), a, max(a, b - 1))
                    px = xs[idx]
                    n = int(((px >= lo) & (px <= hi) & np.isfinite(px) & (na[idx] == 1)).sum())
        counts.append(n)
    return counts


def spread_across_experiments(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    """Take n traverses one per experiment, cycling -- not n from the biggest one."""
    order = pool.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
    return pool.assign(rank=order).sort_values(["rank", "exp"]).head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-cell", type=int, default=15,
                        help="traverses per direction x (calls / quiet) cell, or in total "
                             "when --position-band selects them")
    parser.add_argument("--channels", default="1",
                        help="raw mic channels to draw, 0-based, comma separated. '1,0' "
                             "puts the tunnel mic above the nest mic so the localiser's "
                             "evidence is visible on the card.")
    parser.add_argument("--localiser-marks", action="store_true",
                        help="add a ribbon marking each call tunnel-origin or nest-origin")
    parser.add_argument("--arena-frame", action="store_true",
                        help="weld an arena_1 (video_center) frame from the moment of "
                             "ENTRY onto the left of each card, so you can see who was "
                             "left outside and therefore is not the traveller")
    parser.add_argument("--nest-frame", action="store_true",
                        help="weld a nest_top frame from the moment of entry onto each "
                             "card, so you can see who was already in the nest")
    parser.add_argument("--prior-nest", choices=("any", "yes", "no"), default="any",
                        help="'no' keeps only traverses where the NEST was silent in the "
                             "seconds before entry -- the condition that separates the "
                             "arrival burst from an ongoing colony bout")
    parser.add_argument("--prior-window", type=float, default=5.0)
    parser.add_argument("--localiser-quantile", type=float, default=0.99)
    parser.add_argument("--direction", help="keep only this direction")
    parser.add_argument("--keep-capped", action="store_true",
                        help="keep traverses whose tunnel never emptied within "
                             "MAX_LINGER_S. Their t_out is invented (t_exit + 5 s), so the "
                             "'out of tunnel' line sits well after the animal has left. "
                             "Dropped by default.")
    parser.add_argument("--select-csv",
                        help="CSV of exp,file_num,t_entry naming exactly which traverses "
                             "to draw; any extra column is printed on the card. Skips the "
                             "usable-threshold filter, since an explicit request should "
                             "not be silently dropped for having no localiser.")
    parser.add_argument("--position-band", help="lo,hi -- select traverses whose calls fall in "
                                                "this stretch of tunnel, e.g. 0.05,0.15")
    parser.add_argument("--min-in-band", type=int, default=2,
                        help="how many calls inside the band a traverse needs to qualify")
    parser.add_argument("--per-sheet", type=int, default=6, help="crossings per JPG (default 6)")
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--max-width", type=int, default=6000,
                        help="start a new sheet rather than let it get wider than this "
                             "(default 6000 px); a single card wider than the budget still "
                             "gets its own sheet")
    parser.add_argument("--split-calls", action="store_true",
                        help="separate sheets for traverses with and without calls in view")
    parser.add_argument("--sort", choices=("calls", "width"), default="calls",
                        help="order cards within a sheet group. 'width' groups similar-length "
                             "traverses so one long card cannot stretch a whole sheet.")
    parser.add_argument("--align", choices=("left", "right", "auto"), default="auto",
                        help="which edge to line the cards up on. "
                             "'auto' is now left for every direction: to_nest cards "
                             "mirror their frames rather than their time axis, so t=0 is "
                             "at the left edge on every card.")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    channels = tuple(int(c) for c in args.channels.split(","))
    traverses = load_traverses(scan, args.date, single_animal=True,
                               keep_capped=args.keep_capped)
    traverses = traverses[traverses.tile.notna()]

    if args.direction:
        traverses = traverses[traverses.direction == args.direction]
    if args.select_csv:
        traverses = select_from_csv(traverses, Path(args.select_csv))

    origin = None
    if args.prior_nest != "any" or args.localiser_marks:
        # the same localiser split the figures use, so a card set matches a panel
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.analysis.raster_and_rate_tunnel import localised_sides
        tunnel_calls, nest_calls, _ = localised_sides(
            scan, quantile=args.localiser_quantile)
        if args.localiser_marks:
            origin = (tunnel_calls, nest_calls)

    if args.prior_nest != "any":
        want = args.prior_nest == "yes"
        keep = []
        for r in traverses.itertuples():
            times = nest_calls.get((int(r.exp), int(r.file_num)))
            had = bool(times is not None and times.size
                       and ((times >= r.t_entry - args.prior_window)
                            & (times < r.t_entry)).any())
            keep.append(had == want)
        before = len(traverses)
        traverses = traverses[keep]
        print(f"prior-nest={args.prior_nest}: {len(traverses)} of {before} traverses "
              f"(nest {'called' if want else 'silent'} in the "
              f"{args.prior_window:g} s before entry)")

    localised = tunnel_localised(scan)
    if not args.select_csv:
        traverses = traverses[[(e, f) in localised
                               for e, f in zip(traverses.exp, traverses.file_num)]]

    def n_localised(row) -> int:
        times = localised.get((row.exp, row.file_num))
        if times is None or not len(times):
            return 0
        t0, t1 = row.t_entry - BEFORE_S, row.t_out + AFTER_S
        return int(((times >= t0) & (times <= t1)).sum())

    traverses["n_tunnel_calls"] = [n_localised(r) for r in traverses.itertuples()]
    print(f"{len(traverses)} traverses in experiments with a usable threshold; "
          f"{int((traverses.n_tunnel_calls > 0).sum())} have a tunnel-localised call")

    if args.select_csv:
        picked = [(r.direction, int(r.n_tunnel_calls), r) for r in traverses.itertuples()]
        print(f"  drawing {len(picked)} selected traverses from "
              f"{traverses.exp.nunique()} experiments")
    elif args.position_band:
        lo, hi = (float(v) for v in args.position_band.split(","))
        traverses = traverses.assign(n_in_band=count_in_band(scan, traverses, lo, hi))
        qualified = traverses[traverses.n_in_band >= args.min_in_band]
        print(f"{len(qualified)} traverses with >= {args.min_in_band} calls in "
              f"position {lo}-{hi} (of {len(traverses)})")
        order = qualified.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
        chosen = qualified.assign(rank=order).sort_values(
            ["rank", "n_in_band"], ascending=[True, False]).head(args.per_cell)
        picked = [(r.direction, int(r.n_in_band), r) for r in chosen.itertuples()]
        print(f"  drawing {len(picked)} from {chosen.exp.nunique()} experiments")
    else:
        picked = []
        for direction in ("to_arena", "to_nest"):
            for label, mask in (("calls", traverses.n_tunnel_calls > 0),
                                ("quiet", traverses.n_tunnel_calls == 0)):
                pool = traverses[(traverses.direction == direction) & mask]
                if pool.empty:
                    continue
                chosen = spread_across_experiments(pool, args.per_cell)
                picked += [(direction, int(r.n_tunnel_calls), r) for r in chosen.itertuples()]
                print(f"  {direction:<9} {label:<6} {len(chosen):3d} from "
                      f"{chosen.exp.nunique()} experiments")

    das_cache: dict[int, dict] = {}
    drift_cache: dict[tuple[int, int], float] = {}
    cards = []
    for direction, n_calls, row in picked:
        image = build_card(scan, row, direction, channels, das_cache,
                           drift_cache, args.nest_frame, origin, args.arena_frame)
        if image is None:
            continue
        entry = float(row.t_entry)
        text = (f"exp {row.exp}  {row.video}  t={entry:.2f}s  {direction}  "
                f"traverse {row.traverse_s:.1f}s  in tunnel {row.t_out - entry:.1f}s  "
                f"{'%d calls in view' % n_calls if n_calls else 'no calls in view'}")
        extra = "  ".join(f"{c[4:]}={getattr(row, c):.4f}"
                          for c in row._fields if c.startswith("sel_"))
        if extra:
            text += f"   [{extra}]"
        cards.append({"image": image, "label": text, "direction": direction,
                      "calls": n_calls, "exp": int(row.exp), "video": row.video,
                      "t_entry": entry, "t_out": float(row.t_out), "sheet": ""})
    if not cards:
        raise SystemExit("nothing selected -- loosen --min-in-band or widen --position-band")

    written = build_sheets(cards, out_dir / "sheets", args.per_sheet, args.quality,
                           args.align, args.sort, args.max_width, args.split_calls)
    pd.DataFrame([{k: c[k] for k in
                   ("exp", "video", "t_entry", "t_out", "direction", "calls", "sheet")}
                  for c in cards]).to_csv(out_dir / "selection.csv", index=False)
    print(f"\n{len(cards)} crossings -> {len(written)} sheets in {out_dir}/sheets"
          f"\nwrote {out_dir}/selection.csv")


if __name__ == "__main__":
    main()
