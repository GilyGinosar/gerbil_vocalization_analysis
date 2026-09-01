#!/usr/bin/env python
"""Flag code that loads calls or traverses without going through the loaders.

The never-usable rules (truncated last chunk, capped t_out) are enforced as
defaults inside `data_rules.load_traverses` and `ethogram_io.load_all_calls`.
Anything that reaches past them to `pd.read_parquet` gets raw data with the bad
rows still in, and nothing warns you -- which is exactly how the nest-motion
analysis ran for two days on traverses from truncated chunks.

So this greps for the bypass. Run it before committing, or wire it into CI:

    python scripts/utils/check_direct_reads.py            # report
    python scripts/utils/check_direct_reads.py --strict   # exit 1 if any found

Notebooks are checked too, since that is where most exploration happens and
where a rule in a README is least likely to be read.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# reading these files directly bypasses the rules
PATTERNS = [
    (re.compile(r"read_parquet\s*\([^)]*traverses_"), "traverses parquet",
     "use data_rules.load_traverses()"),
    (re.compile(r"read_parquet\s*\([^)]*all_calls"), "pooled calls parquet",
     "use ethogram_io.load_all_calls()"),
    (re.compile(r"read_csv\s*\([^)]*all_calls"), "pooled calls csv",
     "use ethogram_io.load_all_calls()"),
]

# the loaders themselves, and code whose job is to audit the raw data
ALLOWED = {
    "scripts/utils/data_rules.py",
    "scripts/utils/ethogram_io.py",
    "scripts/utils/check_direct_reads.py",
    "scripts/pipeline/pool_calls.py",
}


def code_of(path: Path) -> list[tuple[int, str]]:
    """Lines of a .py file, or of a notebook's code cells."""
    if path.suffix == ".py":
        return list(enumerate(path.read_text(errors="ignore").splitlines(), 1))
    try:
        nb = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return []
    out, n = [], 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in cell.get("source", []):
            n += 1
            out.append((n, line.rstrip("\n")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any are found")
    ap.add_argument("--include-archive", action="store_true",
                    help="also check scripts/archive, which is frozen by convention")
    args = ap.parse_args()

    hits = []
    for path in sorted([*ROOT.glob("scripts/**/*.py"), *ROOT.glob("notebooks/*.ipynb")]):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        if not args.include_archive and "/archive/" in rel:
            continue
        for lineno, line in code_of(path):
            if line.lstrip().startswith("#"):
                continue
            for rx, what, fix in PATTERNS:
                if rx.search(line):
                    hits.append((rel, lineno, what, fix, line.strip()[:80]))

    if not hits:
        print("no direct reads — everything goes through the loaders")
        return
    print(f"{len(hits)} direct read(s) bypassing the never-usable rules:\n")
    for rel, lineno, what, fix, snippet in hits:
        print(f"  {rel}:{lineno}  [{what}]\n      {snippet}\n      -> {fix}")
    if args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
