#!/usr/bin/env python
"""Tick the nests that are not actually empty, in a browser.

`nest_motion.py` scores the 5 s before a `to_nest` arrival and calls a traverse
"still" when nothing moved. Still is not the same as empty: a gerbil asleep under
the bedding moves less than the sensor noise floor and scores exactly zero. The
two readings support opposite conclusions about who is calling, and no statistic
separates them -- but an eye on the frames does, in a couple of minutes.

So this lays every still traverse out as a strip of frames from the pre-entry
window and asks one question per row: is anybody in there? Tick the ones that are
NOT empty, download the CSV, and the analysis can then split still-and-empty from
still-but-occupied instead of pooling them.

Frames are read sequentially from one seek per traverse, the same way the score
was measured -- seeking to individual frames snaps to keyframes and silently
returns the same picture several times over.

    python scripts/video/nest_empty_picker.py \
        --motion-csv exports/burrow/nest_motion/nest_motion.csv \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir exports/burrow/nest_empty_picker

Then open index.html, tick, and hit "Download picks.csv".
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.nest_motion import CHANGED, FPS, SMALL  # noqa: E402
from scripts.utils.data_rules import load_traverses  # noqa: E402
from scripts.pipeline.paths import BASE_RAW  # noqa: E402

PANEL_H = 260           # px per frame in the strip
JPEG_Q = 82
FONT = cv2.FONT_HERSHEY_SIMPLEX


def strip_for(exp: int, file_num: int, t_entry: float, pre: float,
              n_frames: int, fast: bool = False,
              panel_h: int = PANEL_H) -> np.ndarray | None:
    """A row of frames spanning [t_entry - pre, t_entry].

    Two modes, and the difference is 5x of runtime over a thousand traverses:

    `fast` seeks to each display frame. Seeking snaps to the nearest keyframe, so
    the frame returned may be a few frames off the one asked for -- irrelevant when
    the question is "is there an animal in this picture", and fatal only if you
    then DIFF two seeked frames, which is why the slow path exists.

    The default decodes the whole window in sequence so it can also measure the
    peak frame-to-frame change. That number is worth having on a small sheet; on a
    large one it is redundant, since `motion_pre` for the same window is already in
    the score table and goes on the label either way.
    """
    if fast:
        return _strip_seek(exp, file_num, t_entry, pre, n_frames, panel_h)
    path = (BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
            / f"video_nest_top_{file_num:03d}.mp4")
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    start = max(int((t_entry - pre) * FPS), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    total = max(int(pre * FPS), 1)
    want = {int(round(i)) for i in np.linspace(0, total - 1, n_frames)}
    frames, prev, changed = [], None, []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SMALL)
        if prev is not None:
            changed.append(float((cv2.absdiff(small, prev) > CHANGED).mean()))
        prev = small
        if i in want:
            up = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            scale = panel_h / up.shape[0]
            up = cv2.resize(up, (max(int(up.shape[1] * scale), 1), panel_h))
            cv2.putText(up, f"{i / FPS - pre:+.0f}", (3, 12), FONT, 0.34,
                        (120, 235, 255), 1, cv2.LINE_AA)
            frames.append(up)
    cap.release()
    if not frames:
        return None
    peak = max(changed) if changed else 0.0
    strip = cv2.hconcat(frames)
    if panel_h >= 200:
        cv2.putText(strip, f"peak frame-to-frame change: {peak:.4f}",
                    (5, strip.shape[0] - 8), FONT, 0.44, (150, 150, 150), 1, cv2.LINE_AA)
    return strip


def _strip_seek(exp: int, file_num: int, t_entry: float, pre: float,
                n_frames: int, panel_h: int = PANEL_H) -> np.ndarray | None:
    """One seek per displayed frame; no differencing, so keyframe snap is benign."""
    path = (BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
            / f"video_nest_top_{file_num:03d}.mp4")
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    frames = []
    for off in np.linspace(-pre, 0, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int((t_entry + off) * FPS), 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        up = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        scale = panel_h / up.shape[0]
        up = cv2.resize(up, (max(int(up.shape[1] * scale), 1), panel_h))
        cv2.putText(up, f"{off:+.0f}", (3, 12), FONT, 0.34,
                    (120, 235, 255), 1, cv2.LINE_AA)
        frames.append(up)
    cap.release()
    return cv2.hconcat(frames) if frames else None


HEAD = """<!doctype html><meta charset=utf-8><title>is the nest actually empty?</title>
<style>
 body{font:13px system-ui;margin:0;background:#1a1526;color:#eee}
 #bar{position:sticky;top:0;background:#2a2140;padding:10px 14px;display:flex;gap:12px;
      align-items:center;flex-wrap:wrap;z-index:9}
 button{font:14px system-ui;padding:6px 14px;background:#7c5cff;color:#fff;border:0;
        border-radius:6px;cursor:pointer}
 .card{background:#241d38;border-radius:6px;padding:8px;margin:10px 14px;display:block;
       cursor:pointer}
 .card img{width:100%;display:block;border-radius:4px}
 .card span{display:block;font-size:14px;color:#cdc;margin-top:6px}
 .card:has(:checked){outline:3px solid #ff6b6b;background:#3a2030}
 #picks{width:calc(100% - 28px);height:90px;margin:0 14px;display:none;background:#120f1c;
        color:#f99;border:1px solid #444}
 .hint{font-size:12px;color:#aaa}
</style>
<div id=bar>
 <b>marked NOT empty: <span id=n>0</span> across all pages</b>
 <span class=hint>this page: <span id=pageN>0</span></span>
 <button onclick="show()">Show picks (copy the box)</button>
 <button onclick="dl()">Download picks.csv</button>
 <span class=hint>Tick every nest where you can see an animal. Ticked = NOT empty.
  Picks are kept in this browser, so you can close the tab and come back.</span>
 <span id=nostore style="display:none;color:#ffb4b4">storage unavailable (file:// ?) —
  picks will NOT be remembered and will NOT carry across pages</span>
 <span id=nav class=hint></span>
</div>
<textarea id=picks readonly></textarea>
"""

TAIL = r"""
<script>
 const KEY='__STORAGE_KEY__';
 const boxes=()=>[...document.querySelectorAll('input[type=checkbox]')];
 // Picks live in localStorage as the union across every page, not in this page's
 // DOM: the set is split into pages so a browser is not asked to parse 130 MB of
 // inline images at once, and an export that only saw the current page would
 // silently drop the rest of the work.
 let storageOK=true;
 const warn=()=>{if(!storageOK)nostore.style.display='inline';};
 const load=()=>{try{return new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));}
   catch(e){storageOK=false;return new Set();}};
 const store=set=>{try{localStorage.setItem(KEY,JSON.stringify([...set]));}
   catch(e){storageOK=false;warn();}};
 let picked=load(); warn();
 boxes().forEach(b=>{if(picked.has(b.dataset.id))b.checked=true;});
 const here=()=>boxes().filter(b=>b.checked).length;
 const count=()=>{n.textContent=picked.size;
   pageN.textContent=here()+' / '+boxes().length;};
 const csvText=()=>"exp,file_num,t_entry\n"+[...picked].sort().join("\n")+"\n";
 document.addEventListener('change',e=>{
   const b=e.target; if(!b.dataset||!b.dataset.id)return;
   b.checked?picked.add(b.dataset.id):picked.delete(b.dataset.id);
   store(picked); count(); if(picks.style.display!='none')picks.value=csvText();});
 function show(){picks.style.display='block';picks.value=csvText();picks.select();}
 function dl(){const a=document.createElement('a');
   a.href=URL.createObjectURL(new Blob([csvText()],{type:'text/csv'}));
   a.download='picks.csv';a.click();}
 count();
</script>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion-csv", required=True)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--max-motion", type=float,
                        help="keep traverses at or below this pre-entry motion "
                             "(default: the median of the table, i.e. the 'still' half)")
    parser.add_argument("--pre", type=float, default=5.0,
                        help="seconds before entry; must match the nest_motion run")
    parser.add_argument("--frames", type=int, default=4,
                        help="frames per row. 1 is enough to answer 'is anybody in "
                             "there' and keeps a 1,700-row set to a size a browser "
                             "will actually open")
    parser.add_argument("--fast", action="store_true",
                        help="seek to each displayed frame instead of decoding the "
                             "whole window; ~5x quicker and safe here because nothing "
                             "is differenced")
    parser.add_argument("--page-size", type=int, default=0,
                        help="rows per HTML page; 0 puts everything in index.html. "
                             "Picks are stored as the union across pages, so paging "
                             "does not split the export.")
    parser.add_argument("--exclude-csv", action="append", default=[],
                        help="CSV of exp,file_num,t_entry already scored; repeatable")
    parser.add_argument("--panel-h", type=int, default=PANEL_H,
                        help=f"px per frame (default {PANEL_H}). Drop it when asking "
                             f"for many frames per row, or the row ends up taller than "
                             f"the screen and you scroll more, not less.")
    parser.add_argument("--storage-key", default="nest_empty_picks",
                        help="localStorage key for this picker's verdicts. Give each "
                             "picker set its own: the key is per-ORIGIN, so two sets "
                             "served from the same localhost port otherwise share one "
                             "pick list and each exports the other's rows.")
    parser.add_argument("--rows", default="",
                        help="START:END row range to build, e.g. '350:700'. Lets one "
                             "page be finished and usable while the rest still builds.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1,
                        help="split the build across N concurrent processes. Worth it "
                             "even on one core: building a row is I/O bound on ceph "
                             "(the process sits at ~0%% CPU), so shards overlap each "
                             "other's waits. Each shard writes card fragments; "
                             "--assemble then stitches them into paginated pages.")
    parser.add_argument("--assemble", action="store_true",
                        help="skip building and just page up the fragments on disk")
    args = parser.parse_args()

    m = pd.read_csv(args.motion_csv)
    cut = args.max_motion if args.max_motion is not None else m.motion_pre.median()

    # a traverse whose t_out was invented has an unreliable arrival count, and the
    # count is on the label -- drop them here too so the page agrees with the figures
    tv = load_traverses(args.scan, args.date, keep_capped=True, quiet=True)
    capped = {(int(r.exp), int(r.file_num), round(float(r.t_entry), 2))
              for r in tv[tv.still_in_tunnel_at_cap].itertuples()}
    keep = m[(m.motion_pre <= cut)
             & [(int(r.exp), int(r.file_num), round(float(r.t_entry), 2)) not in capped
                for r in m.itertuples()]]
    if args.exclude_csv:
        seen = set()
        for path in args.exclude_csv:
            done = pd.read_csv(path)
            seen |= {(int(r.exp), int(r.file_num), round(float(r.t_entry), 3))
                     for r in done.itertuples()}
        before = len(keep)
        keep = keep[[(int(r.exp), int(r.file_num), round(float(r.t_entry), 3)) not in seen
                     for r in keep.itertuples()]]
        print(f"excluded {before - len(keep)} already-scored traverses")
    keep = keep.sort_values("motion_pre")
    print(f"{len(keep)} still traverses at motion_pre <= {cut:.4f} "
          f"(of {len(m)} measured); building strips")

    frag_dir = Path(args.out_dir) / "frags"
    frag_dir.mkdir(parents=True, exist_ok=True)
    rows = list(keep.itertuples())

    if not args.assemble:
        lo_r, hi_r = 0, len(rows)
        if args.rows:
            a, _, b = args.rows.partition(":")
            lo_r, hi_r = int(a or 0), int(b or len(rows))
        mine = [(i, r) for i, r in enumerate(rows)
                if lo_r <= i < hi_r and i % args.shards == args.shard]
        print(f"shard {args.shard + 1}/{args.shards}: {len(mine)} rows", flush=True)
        for n, (i, r) in enumerate(mine, 1):
            frag = frag_dir / f"{i:05d}.html"
            if frag.exists():
                continue
            strip = strip_for(int(r.exp), int(r.file_num), float(r.t_entry),
                              args.pre, args.frames, args.fast, args.panel_h)
            ok, buf = (False, None) if strip is None else cv2.imencode(
                ".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if not ok:
                frag.write_text("")      # a placeholder keeps the shard restartable
                continue
            uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
            ident = f"{int(r.exp)},{int(r.file_num)},{float(r.t_entry):.4f}"
            label = (f"exp {int(r.exp)} · file {int(r.file_num)} · t_entry "
                     f"{r.t_entry:.2f}s · motion_pre {r.motion_pre:.4f} · "
                     f"{int(r.arrival_calls)} calls at arrival")
            frag.write_text(f'<label class="card"><img src="{uri}" loading="lazy">'
                            f'<span><input type="checkbox" data-id="{ident}"> '
                            f'{label}</span></label>')
            if n % 25 == 0:
                print(f"  shard {args.shard} [{n}/{len(mine)}]", flush=True)
        print(f"shard {args.shard + 1}/{args.shards} done", flush=True)
        if args.shards > 1:
            return

    cards = [t for t in ((frag_dir / f"{i:05d}.html").read_text()
                         if (frag_dir / f"{i:05d}.html").exists() else ""
                         for i in range(len(rows))) if t]
    print(f"assembling {len(cards)} of {len(rows)} rows")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = args.page_size or len(cards)
    pages = [cards[i:i + size] for i in range(0, len(cards), size)] or [[]]
    names = ["index.html" if len(pages) == 1 else f"page_{i + 1:02d}.html"
             for i in range(len(pages))]
    for i, (page, name) in enumerate(zip(pages, names)):
        nav = ""
        if len(pages) > 1:
            links = " ".join(
                f'<b>{j + 1}</b>' if j == i else f'<a href="{nm}" style="color:#9bf">{j + 1}</a>'
                for j, nm in enumerate(names))
            nav = f"<script>nav.innerHTML='page {i + 1} of {len(pages)} &nbsp; {links}'</script>"
        body = "".join(page)
        path = out_dir / name
        path.write_text(HEAD + body + nav
                        + TAIL.replace("__STORAGE_KEY__", args.storage_key))
        print(f"  {path}  ({len(page)} rows, {path.stat().st_size / 1e6:.1f} MB)")
    keep.to_csv(out_dir / "candidates.csv", index=False)
    print(f"\n{len(cards)} traverses over {len(pages)} page(s) in {out_dir}")


if __name__ == "__main__":
    main()
