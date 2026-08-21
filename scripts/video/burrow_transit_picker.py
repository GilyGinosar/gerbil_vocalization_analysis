#!/usr/bin/env python
"""Curate burrow transits on sound + motion together: tunnel spectrogram over video strip.

The tracking repo's ``burrow_transits.py`` finds tunnel crossings from the
``burrow_side`` camera by frame differencing and writes an events CSV (see its
BURROW_HANDOFF.md). Its HTML picker shows a 5-frame video strip per event. This
script rebuilds that picker with the **tunnel microphone's spectrogram stacked
above each strip**, so a crossing can be judged on its acoustic signature too --
and so transits get linked to the calls that DAS detected in the same window.

Each card is one event:

    +--------------------------------------------------+
    | spectrogram, tunnel mic (ch 1), 0.5-45 kHz        |
    | ---- call ribbon: DAS calls, coloured by type ----|
    | frame 0 | frame 1 | frame 2 | frame 3 | frame 4   |
    +--------------------------------------------------+
    [x] video_burrow_side_002.mp4 t=180.4s to_arena ... 3 calls

Two layouts, for two different jobs
-----------------------------------
``--layout fixed`` (default) -- **for judging whether calling belongs to the
crossing**. The spectrogram spans the same number of seconds on every card
(``--context`` each side of the crossing's midpoint), so all cards are directly
comparable and you can see whether a burst is specific to the crossing or was
going on anyway. The crossing itself is shaded and labelled. Crossings longer
than ``--context`` widen their card to hold the whole crossing plus half a
context each side, and say so in the label.

``--layout eventspan`` -- **for judging a single crossing precisely**. The
spectrogram is scaled to the crossing and shares an *exact* time axis with the
frames: frame *k* is sampled at ``start + (end-start)*k/(per-1)`` and drawn FW px
wide, so the tile's centre sits directly above the audio at that frame.
``--pad`` adds context and the strip gets matching black margins so the
alignment holds. The cost is that every card has its own time scale.

Which audio file, which channel (the handoff's blocker, now resolved)
---------------------------------------------------------------------
Video and audio in ``concatenated_data_cam_mic_sync`` are paired by the trailing
file index -- ``video_burrow_side_002.mp4`` goes with ``channel_NN_file_002.wav``
(the folder's own README.txt states the pairing). Each wav is one mono channel,
125 kHz, float32, ~360 s.

Channel numbering is **0-based**: files run ``channel_00`` .. ``channel_20``, and
for experiments >= 272 ``audio_processing_config.get_channel_mapping`` wires raw
channels {0,1} to "underground", {2,3} to arena_1, {4,5} to arena_2. So the
handoff's "tunnel mic = channel 1" is ``channel_01_file_NNN.wav``. Measured
against the 52 curated transits of experiment 492, in-transit RMS rises +7.2 dB
on ch01 versus +2.9 dB on its nest-pair partner ch00 and ~0 dB on all four arena
mics -- ch01 is the tunnel mic.

Clock drift
-----------
The wav and the mp4 are written by different clocks, so a file pair's durations
disagree (~0.07% on experiment 237). Audio reads are scaled by
``audio_dur / video_dur`` per file, the same correction
``sync_video_spectrogram.py`` makes. On experiment 492 that ratio is 1.0000025
(0.9 ms over 360 s), i.e. this dataset barely drifts -- but the correction costs
nothing and other experiments need it.

Usage
-----
    # picker for one experiment's events CSV
    python scripts/video/burrow_transit_picker.py \
        --from-csv .../transits_492_curated.csv --exp 492 --out-dir picker_492

    # the exact-shared-axis layout instead, on the full uncurated event list
    python scripts/video/burrow_transit_picker.py \
        --from-csv .../transits_492.csv --exp 492 --out-dir picker_492_all \
        --layout eventspan --pad 1.0

Tick the good crossings in the HTML, hit "Download picks.csv", then filter with
the tracking repo's unchanged round-trip:

    python burrow_transits.py --apply-picks picks.csv \
        --from-csv transits_492.csv --out transits_492_curated.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pipeline.audio_processing_config import get_channel_mapping  # noqa: E402
from scripts.pipeline.paths import BASE_RAW, experiment_audio_dir  # noqa: E402

# --- geometry ---------------------------------------------------------------
FPS = 30           # burrow_side cameras run at 30 fps
PER = 5            # video frames sampled across each event (the strip)
FW, FH = 380, 120  # px per frame tile
SPEC_H = 170       # px height of the spectrogram
RIBBON_H = 14      # px height of the DAS call ribbon under the spectrogram
DEF_ROI = (600, 470, 1520, 760)   # tunnel box in the 1600x1200 frame; per-camera-framing
GOP_FRAMES = 250   # keyframe spacing (~8.3 s); a seek costs about this much decoding
CONTEXT_S = 10.0   # "fixed" layout: seconds of audio shown each side of the crossing midpoint

# --- spectrogram ------------------------------------------------------------
FMIN, FMAX = 500, 45000   # Hz. Measured on experiment 492: DAS calls peak at
                          # 20-30 kHz with content 10-40 kHz, while the movement
                          # itself (rustle, scratching) sits below 10 kHz. This
                          # band shows both -- rustle along the bottom, calls above.
NFFT, HOP = 512, 128      # at 125 kHz: 4.1 ms window, 1.0 ms hop
DRANGE_DB = 48            # displayed dynamic range below the window's peak; wider than
                          # this lifts the noise floor into magma's purples and speckles the card

TUNNEL_CHANNEL = 1        # raw mic channel; see the module docstring

# DAS event_type -> BGR, for the call ribbon
CALL_COLORS = {
    "alarm":     (60, 60, 255),
    "high-freq": (0, 170, 255),
    "newborn":   (80, 220, 80),
    "stacks":    (255, 160, 60),
    "warble":    (220, 100, 220),
    "noise":     (150, 150, 150),
}


def file_index(video_name: str) -> int:
    """2 from 'video_burrow_side_002.mp4' -- the index that pairs video with audio."""
    match = re.search(r"_(\d+)\.mp4$", video_name)
    if not match:
        raise ValueError(f"cannot read a file index out of {video_name!r}")
    return int(match.group(1))


def audio_path(datadir: Path, channel: int, index: int) -> Path:
    """The wav for one channel of one file index, across both naming schemes.

    Modern dumps (e.g. experiment 492) use ``channel_01_file_002.wav``; older
    ones use ``channel_1_2.wav``. Try every padding combination, as
    ``sync_video_spectrogram`` does.
    """
    for chan_width in (2, 1):
        for infix in ("_file_", "_"):
            for num_width in (3, 4, 2, 1):
                path = datadir / f"channel_{channel:0{chan_width}d}{infix}{index:0{num_width}d}.wav"
                if path.is_file():
                    return path
    raise FileNotFoundError(f"no channel {channel} wav for file index {index} in {datadir}")


def load_calls(exp: int) -> dict[int, list[tuple[float, float, str]]]:
    """DAS calls for one experiment, grouped by file index.

    Returns {file_index: [(start_s, stop_s, event_type), ...]} in AUDIO-file
    seconds. Only underground calls are kept -- the tunnel mic belongs to the
    underground pair, so arena calls would just be crosstalk on this strip.
    """
    path = experiment_audio_dir(exp) / "calls.csv"
    if not path.exists():
        print(f"no calls.csv at {path} -- continuing without the call ribbon")
        return {}
    calls: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["assigned_location"] != "underground":
                continue
            calls[int(row["file_num"])].append(
                (float(row["start_time_file_sec"]),
                 float(row["stop_time_file_sec"]),
                 row["event_type"]))
    return calls


def video_duration(cap: cv2.VideoCapture) -> float:
    """Seconds of video, from the frame count and the reported fps."""
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    return frames / fps


def fixed_window(start_s: float, end_s: float, context_s: float) -> tuple[float, float]:
    """The same span of seconds on every card, centred on the crossing.

    Crossings run from 0.7 s to 31 s, so an event-scaled card silently changes
    its time axis from one crossing to the next and 52 of them cannot be
    compared by eye. A fixed window fixes the scale and shows the quiet (or
    not) seconds either side, which is what tells you whether calling belongs
    to the crossing or was going on anyway.
    """
    middle = (start_s + end_s) / 2
    # A crossing longer than the window would run off both edges of its own card,
    # so you would never see it start or end. Those cards widen to hold the whole
    # crossing plus half a context each side -- their scale differs, and the label
    # says so. Crossings up to context_s long (42 of experiment 492's 52) keep the
    # common scale.
    half = max(context_s, (end_s - start_s) / 2 + context_s / 2)
    return middle - half, middle + half


def mark_crossing(tile: np.ndarray, t0: float, t1: float, start_s: float, end_s: float,
                  context_s: float) -> None:
    """Shade the crossing inside a fixed window and put a lag scale under it."""
    width = tile.shape[1]
    xa = max(0, min(width, int(width * (start_s - t0) / (t1 - t0))))
    xb = max(xa + 1, min(width, int(width * (end_s - t0) / (t1 - t0))))
    lifted = tile[:, xa:xb].astype(np.int16) + np.array([30, 10, 0], np.int16)
    tile[:, xa:xb] = np.clip(lifted, 0, 255).astype(np.uint8)
    cv2.putText(tile, "crossing starts", (xa + 4, SPEC_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 255, 150), 1)
    cv2.putText(tile, "ends", (xb + 4, SPEC_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 255, 150), 1)
    middle = (start_s + end_s) / 2
    for lag in (-context_s / 2, context_s / 2):
        x = int(width * (middle + lag - t0) / (t1 - t0))
        cv2.line(tile, (x, SPEC_H - 10), (x, SPEC_H), (170, 170, 170), 1)
        cv2.putText(tile, f"{lag:+.0f}s", (x + 3, SPEC_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 170), 1)


def event_window(start_s: float, end_s: float, pad_s: float) -> tuple[float, float, int]:
    """The card's time window, and how many px of black margin the strip needs.

    The strip's own axis runs from the centre of the first tile (t=start_s) to
    the centre of the last (t=end_s), so half a tile of time already hangs off
    each end. ``pad_s`` adds more, as extra pixels outside the frame tiles.
    """
    span = max(end_s - start_s, 1.0 / FPS)
    half_tile_s = span / (2 * (PER - 1))          # time worth half a frame tile
    sec_per_px = span / ((PER - 1) * FW)
    margin_px = int(min(pad_s / sec_per_px, 0.5 * PER * FW))   # cap runaway margins
    t0 = start_s - half_tile_s - margin_px * sec_per_px
    t1 = end_s + half_tile_s + margin_px * sec_per_px
    return t0, t1, margin_px


def read_window(wav: Path, t0: float, t1: float, drift: float, av_offset: float) -> tuple[np.ndarray, float]:
    """Mono audio covering VIDEO-time [t0, t1], zero-padded past the file edges.

    Audio-file time is ``video_time * drift + av_offset``; the returned sample
    rate is the drift-corrected one, so an STFT over these samples comes out on
    the video's time axis directly.
    """
    with sf.SoundFile(str(wav)) as handle:
        fs = handle.samplerate
        a = int(round((t0 * drift + av_offset) * fs))
        b = int(round((t1 * drift + av_offset) * fs))
        lead = max(0, -a)
        handle.seek(max(0, a))
        data = handle.read(max(0, b - max(0, a)), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    tail = (b - a) - lead - len(data)
    if lead or tail > 0:
        data = np.concatenate([np.zeros(lead, np.float32), data, np.zeros(max(0, tail), np.float32)])
    return data, fs * drift


def spectrogram_tile(audio: np.ndarray, fs: float, width: int) -> np.ndarray:
    """A magma BGR image of the audio's spectrogram, exactly ``width`` px wide."""
    if audio.size < NFFT:
        return np.zeros((SPEC_H, width, 3), np.uint8)
    freqs, _, spec = stft(audio, fs=fs, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    band = (freqs >= FMIN) & (freqs <= FMAX)
    power_db = 20 * np.log10(np.abs(spec[band]) + 1e-10)
    ceiling = power_db.max()
    scaled = (power_db - (ceiling - DRANGE_DB)) / DRANGE_DB
    image = np.clip(scaled, 0, 1)
    image = np.flipud(image)                       # low frequencies at the bottom
    image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, (width, SPEC_H), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)


def annotate_spectrogram(tile: np.ndarray, t0: float, t1: float,
                         start_s: float, end_s: float) -> None:
    """Draw the kHz scale and mark where the motion event begins and ends."""
    for khz in (10, 20, 30, 40):
        hz = khz * 1000
        if not FMIN <= hz <= FMAX:
            continue
        y = int(SPEC_H * (1 - (hz - FMIN) / (FMAX - FMIN)))
        cv2.line(tile, (0, y), (18, y), (200, 200, 200), 1)
        cv2.putText(tile, f"{khz}", (22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
    cv2.putText(tile, "kHz", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
    for t in (start_s, end_s):
        x = int(tile.shape[1] * (t - t0) / (t1 - t0))
        cv2.line(tile, (x, 0), (x, SPEC_H), (120, 255, 120), 1)


def call_ribbon(calls: list[tuple[float, float, str]], t0: float, t1: float,
                width: int, drift: float, av_offset: float) -> tuple[np.ndarray, dict[str, int]]:
    """A thin bar per DAS call in the window, coloured by type; also counts them.

    Call times are audio-file seconds, so they come back to video time as
    ``(t - av_offset) / drift`` before being placed on the shared axis.
    """
    ribbon = np.full((RIBBON_H, width, 3), 26, np.uint8)
    counts: dict[str, int] = defaultdict(int)
    for start_audio, stop_audio, event_type in calls:
        start = (start_audio - av_offset) / drift
        stop = (stop_audio - av_offset) / drift
        if stop < t0 or start > t1:
            continue
        counts[event_type] += 1
        xa = int(width * (start - t0) / (t1 - t0))
        xb = max(xa + 2, int(width * (stop - t0) / (t1 - t0)))
        cv2.rectangle(ribbon, (xa, 2), (xb, RIBBON_H - 3),
                      CALL_COLORS.get(event_type, (200, 200, 200)), -1)
    return ribbon, dict(counts)


def strip_frame_numbers(start_s: float, end_s: float) -> list[int]:
    """The PER frame numbers sampled evenly across one event."""
    first, last = int(start_s * FPS), int(end_s * FPS)
    return [first + (last - first) * slot // max(1, PER - 1) for slot in range(PER)]


def read_frames(cap: cv2.VideoCapture, wanted: list[int]) -> dict[int, np.ndarray]:
    """Decode just the wanted frames, in one forward pass over the video.

    Seeking is expensive here: keyframes are ~8 s (250 frames) apart and the
    files live on ceph, so ``cap.set(POS_FRAMES)`` silently decodes up to a
    whole GOP -- about as much work as simply grabbing forward. So we walk
    forward with ``grab()`` (decode, skip the colour conversion) whenever the
    next wanted frame is less than a GOP ahead, and only seek across the bigger
    gaps between events.
    """
    frames: dict[int, np.ndarray] = {}
    position = 0        # index of the frame the next read() will return
    for target in sorted(set(wanted)):
        if target < position or target - position > GOP_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            position = target
        while position < target:
            if not cap.grab():
                return frames
            position += 1
        ok, frame = cap.read()
        position += 1
        if ok:
            frames[target] = frame
    return frames


def frame_strip(frames: dict[int, np.ndarray], numbers: list[int],
                roi: tuple[int, int, int, int], margin_px: int,
                gray: bool = True) -> np.ndarray:
    """The PER-frame strip, with black margins so it lines up with the spectrogram.

    These cameras are IR-illuminated with no IR-cut correction, so the raw frames
    come out heavily blue/magenta (mean BGR of the tunnel ROI is about
    172/105/134). That tint carries no information and fights the magma
    spectrogram above, so tiles are desaturated by default; ``gray=False`` keeps
    the camera's own colour.
    """
    x1, y1, x2, y2 = roi
    tiles = []
    for frame_no in numbers:
        frame = frames.get(frame_no)
        if frame is None:
            tiles.append(np.zeros((FH, FW, 3), np.uint8))
            continue
        tile = cv2.resize(frame[y1:y2, x1:x2], (FW, FH))
        if gray:
            tile = cv2.cvtColor(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, f"t={frame_no / FPS:.1f}s", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tiles.append(tile)
    strip = cv2.hconcat(tiles)
    if margin_px:
        margin = np.zeros((FH, margin_px, 3), np.uint8)
        strip = cv2.hconcat([margin, strip, margin])
    return strip


def build_cards(events_csv: Path, datadir: Path, roi, pad_s: float, drift_on: bool,
                av_offset: float, channel: int, calls_by_file, skip_unclear: bool,
                gray_video: bool = True, layout: str = "fixed",
                context_s: float = CONTEXT_S):
    """One card (direction, id, image data-URI, label) per event, video by video."""
    rows = [r for r in csv.DictReader(open(events_csv))
            if not (skip_unclear and r["direction"] == "unclear")]
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)

    cards = []
    for video_name in sorted(by_video):
        video = datadir / video_name
        if not video.exists():
            print(f"missing {video}, skipping its {len(by_video[video_name])} events")
            continue
        index = file_index(video_name)
        wav = audio_path(datadir, channel, index)
        cap = cv2.VideoCapture(str(video))

        with sf.SoundFile(str(wav)) as handle:
            audio_dur = handle.frames / handle.samplerate
        drift = audio_dur / video_duration(cap) if drift_on else 1.0

        # decode every frame this video needs in one forward pass, then cut cards
        numbers = {row["start_s"]: strip_frame_numbers(float(row["start_s"]), float(row["end_s"]))
                   for row in by_video[video_name]}
        frames = read_frames(cap, [n for ns in numbers.values() for n in ns])
        cap.release()

        for row in by_video[video_name]:
            start_s, end_s = float(row["start_s"]), float(row["end_s"])
            if layout == "fixed":
                # frames at full size; the spectrogram above spans the same
                # number of seconds on every card, with the crossing shaded
                t0, t1 = fixed_window(start_s, end_s, context_s)
                strip = frame_strip(frames, numbers[row["start_s"]], roi, 0, gray_video)
            else:
                t0, t1, margin_px = event_window(start_s, end_s, pad_s)
                strip = frame_strip(frames, numbers[row["start_s"]], roi, margin_px, gray_video)
            width = strip.shape[1]

            audio, fs = read_window(wav, t0, t1, drift, av_offset)
            spec = spectrogram_tile(audio, fs, width)
            annotate_spectrogram(spec, t0, t1, start_s, end_s)
            if layout == "fixed":
                mark_crossing(spec, t0, t1, start_s, end_s, context_s)
            ribbon, counts = call_ribbon(calls_by_file.get(index, []), t0, t1,
                                         width, drift, av_offset)

            ok, buf = cv2.imencode(".jpg", cv2.vconcat([spec, ribbon, strip]),
                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            call_text = ("  calls: " + ", ".join(f"{n} {t}" for t, n in sorted(counts.items()))
                         if counts else "  no calls")
            scale_note = ""
            if layout == "fixed" and (t1 - t0) > 2 * context_s + 0.01:
                scale_note = f"  [wider card: {t1 - t0:.0f}s, this crossing is long]"
            label = (f"{video_name}  t={row['start_s']}s  {row['direction']}  "
                     f"{end_s - start_s:.1f}s  x{row['x_start']}->{row['x_end']}"
                     f"{call_text}{scale_note}")
            cards.append((row["direction"], f"{video_name}|{row['start_s']}",
                          "data:image/jpeg;base64," + base64.b64encode(buf).decode(), label))
        print(f"{video_name}: {len(by_video[video_name])} cards", flush=True)
    return cards


HTML_HEAD = """<!doctype html><meta charset=utf-8><title>burrow transit picker (sound + motion)</title>
<style>
 body{font:13px system-ui;margin:0;background:#1a1526;color:#eee}
 #bar{position:sticky;top:0;background:#2a2140;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:9}
 button{font:14px system-ui;padding:6px 14px;background:#7c5cff;color:#fff;border:0;border-radius:6px;cursor:pointer}
 h2{margin:16px 14px 6px;text-transform:capitalize}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(900px,1fr));gap:14px;padding:0 14px}
 .card{background:#241d38;border-radius:6px;padding:8px;display:block;cursor:pointer}
 .card img{width:100%;display:block;border-radius:4px;image-rendering:auto}
 .card span{display:block;font-size:14px;color:#cdc;margin-top:6px}
 .card:has(:checked){outline:3px solid #4ade80}
 #picks{width:100%;height:80px;margin:0 14px;display:none;background:#120f1c;color:#9f9;border:1px solid #444}
 .key{font-size:12px;color:#aaa;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:3px;vertical-align:-1px}
</style>
<div id=bar>
 <b>picked: <span id=n>0</span></b>
 <button onclick="show()">Show picks (copy the box)</button>
 <button onclick="dl()">Download picks.csv</button>
 <span>tick the good crossings, then copy or download</span>
 <div class=key>DAS calls (underground):
  <span><i class=sw style="background:#ff3c3c"></i>alarm</span>
  <span><i class=sw style="background:#ffaa00"></i>high-freq</span>
  <span><i class=sw style="background:#50dc50"></i>newborn</span>
  <span><i class=sw style="background:#3ca0ff"></i>stacks</span>
  <span><i class=sw style="background:#dc64dc"></i>warble</span>
  <span style="color:#8f8">| green lines = motion event start/end</span>
 </div>
</div>
<textarea id=picks readonly></textarea>
"""

HTML_TAIL = """
<script>
 const csvText=()=>{const ids=[...document.querySelectorAll('input:checked')].map(c=>c.dataset.id);
   return "video,start_s\\n"+ids.map(i=>i.replace('|',',')).join("\\n")+"\\n";};
 document.addEventListener('change',()=>{n.textContent=document.querySelectorAll('input:checked').length;
   if(picks.style.display!='none')picks.value=csvText();});
 function show(){picks.style.display='block';picks.value=csvText();picks.select();}
 function dl(){const a=document.createElement('a');
   a.href=URL.createObjectURL(new Blob([csvText()],{type:'text/csv'}));a.download='picks.csv';a.click();}
</script>"""


def write_html(cards, out_dir: Path) -> Path:
    """Same self-contained page, same 'video,start_s' picks contract as the handoff."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cards.sort(key=lambda c: (c[0], c[1]))
    body, current = [], None
    for direction, event_id, uri, label in cards:
        if direction != current:
            body.append("</div>" if current is not None else "")
            body.append(f'<h2>{direction}</h2><div class="grid">')
            current = direction
        body.append(f'<label class="card"><img src="{uri}">'
                    f'<span><input type="checkbox" data-id="{event_id}"> {label}</span></label>')
    body.append("</div>")
    path = out_dir / "index.html"
    path.write_text(HTML_HEAD + "".join(body) + HTML_TAIL)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-csv", required=True, help="events CSV from burrow_transits.py")
    parser.add_argument("--exp", type=int, required=True, help="experiment id (finds the raw folder and calls.csv)")
    parser.add_argument("--out-dir", required=True, help="directory to write index.html into")
    parser.add_argument("--datadir", help="override the concatenated_data_cam_mic_sync folder")
    parser.add_argument("--roi", default=",".join(map(str, DEF_ROI)),
                        help="x1,y1,x2,y2 tunnel box; re-check per date with burrow_transits.py --preview")
    parser.add_argument("--channel", type=int, default=TUNNEL_CHANNEL,
                        help=f"raw mic channel, 0-based (default {TUNNEL_CHANNEL} = tunnel)")
    parser.add_argument("--layout", choices=("fixed", "eventspan"), default="fixed",
                        help="fixed: the same time span on every card, crossing shaded -- "
                             "comparable across cards (default). eventspan: the spectrogram "
                             "is scaled to the crossing and shares an exact axis with the frames.")
    parser.add_argument("--context", type=float, default=CONTEXT_S,
                        help=f"fixed layout: seconds shown each side of the crossing "
                             f"(default {CONTEXT_S:.0f})")
    parser.add_argument("--pad", type=float, default=0.5,
                        help="eventspan layout: extra seconds of audio each side (default 0.5)")
    parser.add_argument("--no-calls", action="store_true", help="skip the DAS call ribbon")
    parser.add_argument("--color-video", action="store_true",
                        help="keep the cameras' raw IR colour cast instead of desaturating")
    parser.add_argument("--no-drift", action="store_true", help="skip the audio/video clock-drift correction")
    parser.add_argument("--av-offset", type=float, default=0.0,
                        help="constant seconds added to audio reads after drift correction")
    parser.add_argument("--include-unclear", action="store_true", help="also show 'unclear' events")
    args = parser.parse_args()

    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")
    roi = tuple(int(v) for v in args.roi.split(","))

    mapping = get_channel_mapping(args.exp)
    underground = mapping.get("30", [])
    if args.channel not in underground:
        print(f"note: channel {args.channel} is not in this experiment's underground pair "
              f"{underground} -- the tunnel mic is one of those")

    calls_by_file = {} if args.no_calls else load_calls(args.exp)
    cards = build_cards(Path(args.from_csv), datadir, roi, args.pad, not args.no_drift,
                        args.av_offset, args.channel, calls_by_file,
                        skip_unclear=not args.include_unclear,
                        gray_video=not args.color_video, layout=args.layout,
                        context_s=args.context)
    path = write_html(cards, Path(args.out_dir))
    print(f"picker ready (self-contained): {path}  ({len(cards)} events, ch{args.channel:02d}, "
          f"{args.layout} layout)")


if __name__ == "__main__":
    main()
