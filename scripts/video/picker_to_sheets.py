#!/usr/bin/env python
"""Turn a picker HTML into stacked JPG contact sheets -- for eyeballing over SSH.

`burrow_transit_picker.py` writes a self-contained HTML page, which is the right
format at a desk and the wrong one on a remote shell with no browser. The cards
are already JPEG data-URIs inside that page, so this re-stacks them into a few
big JPGs that open straight in the VS Code editor (or `scp` down) -- no video
re-decoding, seconds instead of ten minutes.

Cards keep their direction grouping, and within each direction the ones with the
most calls in view come first, so the informative crossings are on sheet 1.

    python scripts/video/picker_to_sheets.py \
        --picker exports/burrow_look_492/index.html --out-dir exports/burrow_look_492/sheets
"""
from __future__ import annotations

import argparse
import base64
import html as html_mod
import re
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.video.burrow_transit_picker import CALL_COLORS  # noqa: E402

LABEL_H = 30        # px of caption drawn above each card
LEGEND_H = 34       # px of colour key at the top of every sheet
SEPARATOR_H = 8     # px of dark gap between cards
DIRECTIONS = ["to_arena", "to_nest", "reversal", "unclear"]

CARD_RE = re.compile(
    r'<img src="data:image/jpeg;base64,([^"]+)">\s*<span><input[^>]*data-id="([^"]*)"[^>]*>\s*([^<]+)</span>')


def read_cards(picker_html: Path) -> list[dict]:
    """Every card in the page: its image, its caption, and its direction."""
    page = picker_html.read_text()
    cards = []
    for encoded, event_id, label in CARD_RE.findall(page):
        label = html_mod.unescape(label).strip()
        buffer = np.frombuffer(base64.b64decode(encoded), np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            continue
        direction = next((d for d in DIRECTIONS if f" {d} " in f" {label} "), "other")
        calls = sum(int(n) for n in re.findall(r"(\d+) (?:high-freq|warble|alarm|stacks|newborn)", label))
        cards.append({"image": image, "label": label, "direction": direction,
                      "calls": calls, "event_id": event_id})
    return cards


def legend(width: int, with_calls: bool) -> np.ndarray:
    """The call-type colour key, on every sheet.

    The HTML picker carries this in its top bar; a JPG has no top bar, so
    without it the ribbon ticks are unreadable colours.
    """
    bar = np.full((LEGEND_H, width, 3), 45, np.uint8)
    x = 10
    if with_calls:
        cv2.putText(bar, "call ribbon (underground calls):", (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
        x = 320
        for name, colour in CALL_COLORS.items():
            if name == "noise":
                continue
            cv2.rectangle(bar, (x, 10), (x + 16, 24), colour, -1)
            cv2.putText(bar, name, (x + 22, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (215, 215, 215), 1, cv2.LINE_AA)
            x += 40 + 12 * len(name)
        x += 10
    cv2.putText(bar, "green lines + shading = the traverse   |   scale bar top right = 1 s   |"
                     "   tunnel mic ch01, 0.5-45 kHz", (x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 235, 150), 1, cv2.LINE_AA)
    return bar


def caption(width: int, text: str, calls: int) -> np.ndarray:
    """A dark caption bar; tinted when the card has calls in view."""
    bar = np.full((LABEL_H, width, 3), 34 if calls else 22, np.uint8)
    colour = (120, 235, 255) if calls else (170, 170, 170)
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    return bar


def build_sheets(cards: list[dict], out_dir: Path, per_sheet: int, quality: int,
                 align: str = "left", sort_by: str = "calls",
                 max_width: int = 6000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # a call key only belongs on the sheet when the cards actually carry a ribbon
    with_calls = any("calls" in card["label"] for card in cards)
    written = []

    by_direction: dict[str, list[dict]] = {}
    for card in cards:
        by_direction.setdefault(card["direction"], []).append(card)

    for direction in [d for d in DIRECTIONS + ["other"] if d in by_direction]:
        # Sheet width is set by its widest card, so grouping similar widths keeps
        # short traverses off a page stretched by one long one.
        if sort_by == "width":
            group = sorted(by_direction[direction], key=lambda c: c["image"].shape[1])
        else:
            group = sorted(by_direction[direction], key=lambda c: -c["calls"])
        # Pack cards into sheets: a page closes when it is full OR when adding the
        # next (wider) card would blow the width budget. Sheet width is set by its
        # widest member, so without this one 28 s traverse makes an 18000 px page
        # on which every other card is mostly padding.
        pages, page = [], []
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
            pieces = [legend(width, with_calls)]
            for position, card in enumerate(chunk, start=start + 1):
                image = card["image"]
                if image.shape[1] != width:      # pad, never rescale: rescaling would
                    pad = np.zeros((image.shape[0], width - image.shape[1], 3), np.uint8)
                    # Pad on the side away from t=0 so the anchor column stays put.
                    # Left-to-right cards start at t0, so t=0 is a fixed offset from
                    # the LEFT and they pad right; right-to-left cards are the mirror.
                    image = (cv2.hconcat([pad, image]) if align == "right"
                             else cv2.hconcat([image, pad]))   # change the card's time scale
                pieces.append(caption(width, f"[{position}/{len(group)}] {card['label']}",
                                      card["calls"]))
                pieces.append(image)
                pieces.append(np.full((SEPARATOR_H, width, 3), 60, np.uint8))
            sheet = cv2.vconcat(pieces)
            path = out_dir / f"{direction}_{page_no:02d}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, quality])
            written.append(path)
            print(f"{path}  ({len(chunk)} crossings, {sheet.shape[1]}x{sheet.shape[0]})")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--picker", required=True, help="index.html written by burrow_transit_picker.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-sheet", type=int, default=6, help="crossings per JPG (default 6)")
    parser.add_argument("--quality", type=int, default=88)
    parser.add_argument("--max-width", type=int, default=6000,
                        help="start a new sheet rather than let it get wider than this "
                             "(default 6000 px); a single card wider than the budget still "
                             "gets its own sheet")
    parser.add_argument("--sort", choices=("calls", "width"), default="calls",
                        help="order cards within a direction. 'width' groups similar-length "
                             "traverses together so one long card cannot stretch a whole sheet.")
    parser.add_argument("--align", choices=("left", "right"), default="left",
                        help="which edge to line the cards up on. Use 'right' for cards built "
                             "with --reverse-time, whose t=0 is measured from the right edge; "
                             "otherwise the anchor column drifts card to card.")
    args = parser.parse_args()

    cards = read_cards(Path(args.picker))
    if not cards:
        raise SystemExit(f"no cards found in {args.picker}")
    written = build_sheets(cards, Path(args.out_dir), args.per_sheet, args.quality, args.align, args.sort,
                           args.max_width)
    print(f"\n{len(cards)} crossings -> {len(written)} sheets in {args.out_dir}")


if __name__ == "__main__":
    main()
