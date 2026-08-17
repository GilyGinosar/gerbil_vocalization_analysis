"""Lists the installed entry points. `python main.py` if you forget the names."""


def main():
    print("Installed commands (after `pip install -e .`, on PATH inside .venv):")
    print("  gerbil-average-audio   --experiment-id N | --start-exp A --end-exp B")
    print("  gerbil-rms-assignment  --accepted-calls-dir ... --averaged-wavs-dir ... --output-dir ...")
    print()
    print("Driver scripts (edit the globals at the top, then run):")
    print("  python scripts/pipeline/combine_exp_calls.py       per-experiment calls.csv")
    print("  python scripts/pipeline/run_rms_assignment.py      same, via RMS + dedupe")
    print("  python scripts/pipeline/extract_calls_offline.py   pool a date folder")
    print()
    print("Analysis: scripts/analysis/run_*.py --dates 2026_02   (see README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
