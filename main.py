"""Lists the installed entry points. `python main.py` if you forget the names."""


def main():
    print("Installed commands (after `pip install -e .`, on PATH inside .venv):")
    print("  gerbil-average-audio   --experiment-id N | --start-exp A --end-exp B")
    print("  gerbil-rms-assignment  --accepted-calls-dir ... --averaged-wavs-dir ... --output-dir ...")
    print()
    print("Pipeline, in order (after DAS has run):")
    print("  python scripts/pipeline/combine_exp_calls.py --date-folder 2026_08   -> <exp>/calls.csv")
    print("  python scripts/pipeline/pool_calls.py        --date-folder 2026_08   -> ceph CSV + parquet")
    print()
    print("Video, once the tracking repo has written its CSVs:")
    print("  python scripts/pipeline/pool_detections.py   --date-folder 2026_02   -> detections parquet")
    print("       (also drops the detector's fixed-object detections; --keep-stationary to keep them)")
    print()
    print("  python scripts/pipeline/run_rms_assignment.py   alternative to combine_exp_calls,")
    print("                                                  via RMS + overlap dedupe")
    print()
    print("Analysis: scripts/analysis/run_*.py --dates 2026_02   (see README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
