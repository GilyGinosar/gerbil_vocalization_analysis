#!/usr/bin/env python
"""The current card sheet, in one command, for either direction.

`burrow_cards.py` has accumulated a lot of flags, and the version worth looking
at is one particular combination of eight of them. Remembering which is not a
reasonable ask, and a half-remembered invocation silently produces an older,
worse card -- one with the spectrogram mirrored, or the localiser's discredited
verdict drawn on it. So the combination lives here.

What a card shows, in travel order across the page:

    the compartment the animal LEFT · three spectrograms · the one it ARRIVED in

  * the end panels are video frames at the moment of entry. The one behind the
    animal answers "who stayed put, and so is not the traveller"; the one ahead
    answers "who was waiting". Both flip sides with direction, so the card always
    reads left-to-right in the direction of travel.
  * ARENA_1 / TUNNEL / NEST spectrograms, top to bottom, in the compartment order
    an animal passes through. Arena is the AVERAGED channel 10 -- the signal DAS
    actually scored -- not a raw mic, because arena_1 is ch02 and ch03 and which
    is louder flips between experiments.
  * the DAS ribbon, and the frame strip through the tunnel.

Both directions work. to_nest cards mirror their tiles so travel reads
left-to-right, which is why the panels swap sides; to_arena cards do not.

    python scripts/analysis/make_traverse_cards.py \
        --select-csv data/nest_scoring/cards_empty.csv \
        --out-dir exports/cards/empty

    python scripts/analysis/make_traverse_cards.py --direction to_arena \
        --category active --out-dir exports/cards/active_out
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.utils.publish import publish_many  # noqa: E402

# arena_1 has no camera in these, so the end panel would be a black rectangle
NO_ARENA_VIDEO = {506, 514, 515}
CATEGORY_TABLE = REPO_ROOT / "data" / "nest_scoring" / "nest_category_full.csv"


def selection_from_category(category: str, n: int, out: Path, with_calls: bool,
                            max_in_tunnel: float | None = None,
                            scan: str = "", date: str = "2026_02") -> Path:
    """Pick n traverses of one nest category, one per experiment."""
    cat = pd.read_csv(CATEGORY_TABLE)
    sub = cat[(cat.cat == category) & (~cat.exp.isin(NO_ARENA_VIDEO))]
    if with_calls:
        sub = sub[sub.arrival_calls > 0]
    if max_in_tunnel:
        from scripts.utils.data_rules import load_traverses
        tv = load_traverses(scan, date, single_animal=True, quiet=True)
        dur = {(int(r.exp), int(r.file_num), round(float(r.t_entry), 3)):
               r.t_out - r.t_entry for r in tv.itertuples()}
        keep = [dur.get((int(r.exp), int(r.file_num), round(float(r.t_entry), 3)),
                        1e9) <= max_in_tunnel for r in sub.itertuples()]
        before = len(sub)
        sub = sub[keep]
        print(f"  kept {len(sub)} of {before} at <= {max_in_tunnel:g}s in tunnel")
    if sub.empty:
        raise SystemExit(f"no {category} traverses left after filtering")
    order = sub.sample(frac=1.0, random_state=0).groupby("exp").cumcount()
    pick = sub.assign(rank=order).sort_values(["rank", "exp"]).head(n)
    out.parent.mkdir(parents=True, exist_ok=True)
    pick[["exp", "file_num", "t_entry", "motion_pre", "arrival_calls"]].to_csv(
        out, index=False)
    print(f"{category}: {len(pick)} traverses from {pick.exp.nunique()} experiments")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scan", default="/mnt/ceph/users/gginosar/burrow_scan_2026_02")
    ap.add_argument("--date", default="2026_02")
    ap.add_argument("--direction", choices=("to_nest", "to_arena"), default="to_nest")
    ap.add_argument("--select-csv",
                    help="exp,file_num,t_entry of the traverses to draw")
    ap.add_argument("--category", choices=("empty", "sleeping", "active"),
                    help="instead of --select-csv, sample this nest category")
    ap.add_argument("--n", type=int, default=15, help="traverses when using --category")
    ap.add_argument("--with-calls", action="store_true",
                    help="with --category, keep only traverses that have calls at "
                         "arrival. Selects FOR the phenomenon, so say so when showing "
                         "the result -- the category rate is the unbiased number.")
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--max-in-tunnel", type=float,
                    help="drop traverses longer than this many seconds. Card width "
                         "is 600 px per second, so one 38 s traverse is a 25,000 px "
                         "sheet on which every other card is padding. Filtering to a "
                         "similar length lets --per-sheet actually fill a page.")
    ap.add_argument("--max-width", type=int, default=12000)
    ap.add_argument("--no-localiser-marks", action="store_true",
                    help="drop the tunnel/nest verdict ribbon. It is a POSITION "
                         "gradient, not a compartment split, so it is drawn as "
                         "evidence about the threshold, never as ground truth.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.select_csv:
        select = Path(args.select_csv)
    elif args.category:
        select = selection_from_category(args.category, args.n,
                                         out_dir / "selection_in.csv",
                                         args.with_calls, args.max_in_tunnel,
                                         args.scan, args.date)
    else:
        raise SystemExit("give either --select-csv or --category")

    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts/video/burrow_cards.py"),
           "--scan", args.scan, "--date", args.date,
           "--direction", args.direction,
           "--select-csv", str(select),
           "--channels", "10,1,0",      # arena (averaged) / tunnel / nest
           "--arena-frame", "--nest-frame",
           "--per-sheet", str(args.per_sheet),
           "--sort", "width",
           "--max-width", str(args.max_width),
           "--out-dir", str(out_dir)]
    if not args.no_localiser_marks:
        cmd.append("--localiser-marks")
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        label = args.category or Path(select).stem
        sheets = sorted((out_dir / "sheets").glob("*.jpg"))
        publish_many(sheets, prefix=f"cards_{label}_{args.direction}",
                     date=args.date)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
