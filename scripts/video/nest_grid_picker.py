#!/usr/bin/env python
"""A dense grid for scoring "is anybody in this nest?" -- built for a helper.

One frame per traverse, several traverses per row. One frame is enough to answer
the only question being asked, and a grid means a glance covers five traverses
instead of one, so a few hours of somebody's time covers the whole backlog rather
than a page of it.

Deliberately different from `nest_empty_picker.py` in three ways, all because this
one is opened on somebody else's laptop rather than served from the cluster:

  * ONE self-contained file. No pagination, so there is no cross-page state to
    lose and nothing to serve.
  * The export reads the PAGE, not localStorage. Opened as a file:// URL the
    origin is opaque and storage can be unavailable or silently per-file; the
    work must survive that. Storage is still written when it can be, so a reload
    is recoverable, but nothing depends on it.
  * No call counts on the labels. They are irrelevant to the judgement and would
    make the scoring non-blind to the outcome being tested.

    python scripts/video/nest_grid_picker.py \
        --rows-csv exports/burrow/nest_motion/screen_ranked.csv \
        --out-dir exports/burrow/nest_grid --shards 8
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

from scripts.pipeline.paths import BASE_RAW  # noqa: E402

FPS = 30.0
THUMB_H = 300           # px; the grid scales them down, this is the stored size
JPEG_Q = 78


def thumb(exp: int, file_num: int, t_entry: float, pre: float) -> np.ndarray | None:
    """One frame from the middle of the pre-entry window.

    A single seek: the frame returned may land a few frames off the one asked for
    because seeking snaps to a keyframe, which does not matter when the question
    is "is there an animal in this picture" and nothing is being differenced.
    """
    path = (BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
            / f"video_nest_top_{file_num:03d}.mp4")
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int((t_entry - pre / 2) * FPS), 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    up = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    scale = THUMB_H / up.shape[0]
    return cv2.resize(up, (max(int(up.shape[1] * scale), 1), THUMB_H))


HEAD = """<!doctype html><meta charset=utf-8><title>nest scoring — is anybody home?</title>
<style>
 body{font:13px system-ui;margin:0;background:#14111d;color:#eee}
 #bar{position:sticky;top:0;background:#2a2140;padding:10px 14px;display:flex;gap:14px;
      align-items:center;flex-wrap:wrap;z-index:9;box-shadow:0 2px 8px #0008}
 button{font:14px system-ui;padding:6px 14px;background:#7c5cff;color:#fff;border:0;
        border-radius:6px;cursor:pointer}
 button:hover{background:#9478ff}
 #grid{display:grid;grid-template-columns:repeat(__COLS__,1fr);gap:8px;padding:12px}
 .cell{position:relative;background:#241d38;border-radius:6px;padding:4px;cursor:pointer;
       user-select:none;border:3px solid transparent}
 .cell img{width:100%;display:block;border-radius:3px}
 .cell .id{font-size:10px;color:#8a8398;text-align:center;padding-top:3px}
 .cell.on{border-color:#ff5c5c;background:#3d1f2b}
 .cell.on::after{content:"ANIMAL";position:absolute;top:6px;left:6px;background:#ff5c5c;
   color:#fff;font-size:11px;font-weight:700;padding:1px 6px;border-radius:3px}
 .cell input{display:none}
 #out{width:calc(100% - 28px);height:110px;margin:0 14px 14px;display:none;
      background:#0d0b14;color:#9f9;border:1px solid #444;font:11px monospace}
 .hint{font-size:12px;color:#a9a2b8}
 b{font-size:15px}
</style>
<div id=bar>
 <b>marked ANIMAL: <span id=n>0</span> / <span id=tot>0</span></b>
 <span class=hint>Click every picture where you can see an animal. Click again to undo.</span>
 <button onclick="show()">Show results</button>
 <button onclick="dl()">Download results.csv</button>
 <span id=saved class=hint></span>
</div>
<textarea id=out readonly></textarea>
<div id=grid>
"""

TAIL = r"""</div>
<script>
 const KEY='nest_grid_picks';
 const cells=()=>[...document.querySelectorAll('.cell')];
 tot.textContent=cells().length;
 // The page is the source of truth, not storage: opened as a file:// URL the
 // origin is opaque and localStorage may be unavailable, and three hours of
 // clicking must not depend on it. Storage is best-effort on top.
 let storageOK=true;
 try{
   const saved=new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));
   cells().forEach(c=>{if(saved.has(c.dataset.id))c.classList.add('on');});
 }catch(e){storageOK=false;}
 const marked=()=>cells().filter(c=>c.classList.contains('on'));
 const csvText=()=>"exp,file_num,t_entry\n"+marked().map(c=>c.dataset.id).join("\n")+"\n";
 function persist(){
   try{localStorage.setItem(KEY,JSON.stringify(marked().map(c=>c.dataset.id)));
       saved.textContent='saved in this browser';}
   catch(e){storageOK=false;
     saved.textContent='NOT saved automatically — use Download before closing';
     saved.style.color='#ffb4b4';}
 }
 document.getElementById('grid').addEventListener('click',e=>{
   const c=e.target.closest('.cell'); if(!c)return;
   c.classList.toggle('on');
   n.textContent=marked().length; persist();
   if(out.style.display!='none')out.value=csvText();
 });
 function show(){out.style.display='block';out.value=csvText();out.select();}
 function dl(){const a=document.createElement('a');
   a.href=URL.createObjectURL(new Blob([csvText()],{type:'text/csv'}));
   a.download='results.csv';a.click();}
 n.textContent=marked().length; persist();
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-csv", required=True,
                    help="CSV with exp,file_num,t_entry -- the traverses to score")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--pre", type=float, default=5.0)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    rows = pd.read_csv(args.rows_csv)
    if args.limit:
        rows = rows.head(args.limit)
    rows = rows.reset_index(drop=True)
    out_dir = Path(args.out_dir)
    frag = out_dir / "thumbs"
    frag.mkdir(parents=True, exist_ok=True)

    if not args.assemble:
        mine = [(i, r) for i, r in enumerate(rows.itertuples())
                if i % args.shards == args.shard]
        print(f"shard {args.shard + 1}/{args.shards}: {len(mine)} thumbnails", flush=True)
        for n, (i, r) in enumerate(mine, 1):
            f = frag / f"{i:05d}.txt"
            if f.exists():
                continue
            img = thumb(int(r.exp), int(r.file_num), float(r.t_entry), args.pre)
            if img is None:
                f.write_text("")
                continue
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            f.write_text(base64.b64encode(buf).decode() if ok else "")
            if n % 50 == 0:
                print(f"  shard {args.shard} [{n}/{len(mine)}]", flush=True)
        print(f"shard {args.shard + 1}/{args.shards} done", flush=True)
        if args.shards > 1:
            return

    cards = []
    for i, r in enumerate(rows.itertuples()):
        f = frag / f"{i:05d}.txt"
        if not f.exists():
            continue
        b64 = f.read_text()
        if not b64:
            continue
        ident = f"{int(r.exp)},{int(r.file_num)},{float(r.t_entry):.4f}"
        cards.append(f'<div class="cell" data-id="{ident}">'
                     f'<img src="data:image/jpeg;base64,{b64}">'
                     f'<div class="id">{i + 1}</div></div>')
    html = HEAD.replace("__COLS__", str(args.cols)) + "".join(cards) + TAIL
    path = out_dir / "nest_scoring.html"
    path.write_text(html)
    print(f"wrote {path}  ({len(cards)} of {len(rows)} traverses, "
          f"{path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
