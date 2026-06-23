"""Plot the inter-call gap (ICG) histogram across one or more date folders.

ICG = silent gap between consecutive calls within the same experiment+channel,
measured as next.start - prev.stop (seconds, must be >= 0). Same definition as
bouts.py:icg_s and calc_transitions.collect_inter_call_gaps. Reads from the
all_calls parquet cache built by notebooks/explore_calls_xplatform.ipynb.

This script is the stripped-down sibling of the ICG panel inside
plot_transition_matrices: no CDF overlay, no text annotations on the dashed
threshold lines. Set THRESHOLDS = [] to drop the dashed lines entirely.

Edit the values below and run:  python scripts/plot_icg_histogram.py
"""
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.calc_transitions import collect_inter_call_gaps

# === Edit these before running ============================================
DATE_FOLDERS = ["2025_07"]                # one or more; gaps are pooled across them
N_BINS       = 100                        # geometric (log-spaced) bins
XMAX_SEC     = 1000.0                     # x-axis upper bound; gaps beyond this reported as "N not shown"
THRESHOLDS   = [0.035, 2.0, 300.0]         # dashed vertical lines (no text); set [] to drop entirely
FONT_SIZE    = 15                         # base font size; title/labels/ticks all inherit from this
OUTPUT_NAME  = None                       # None => icg_histogram_<dates_tag>.png
# ==========================================================================


if platform.system() == "Windows":
    PARQUET_DIR = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio\all_calls\parquet_cache")
    OUTPUT_BASE = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    PARQUET_DIR = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio/all_calls/parquet_cache")
    OUTPUT_BASE = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio/combined/transitions/icg")


REQUIRED_COLS = {"event_type", "start_time_experiment_sec", "stop_time_experiment_sec",
                 "channel", "exp", "date_folder"}


def load_calls(date_folders: list[str]) -> pd.DataFrame:
    parts = []
    for date_folder in date_folders:
        path = PARQUET_DIR / f"all_calls_{date_folder}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")
        parts.append(pd.read_parquet(path))
    df = pd.concat(parts, ignore_index=True)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Parquet missing required columns: {sorted(missing)}")
    return df


def gaps_per_experiment(df: pd.DataFrame) -> np.ndarray:
    """Pass one DataFrame per (date_folder, exp) so collect_inter_call_gaps's
    `_source_exp` scoping prevents gaps from bridging across experiments."""
    dfs_by_exp = {f"{date}_{exp}": grp
                  for (date, exp), grp in df.groupby(["date_folder", "exp"])}
    return collect_inter_call_gaps(dfs_by_exp)


def plot_histogram(gaps: np.ndarray, thresholds: list[float],
                   n_bins: int, xmax_sec: float, title: str, out_path: Path) -> None:
    positive = np.asarray(gaps, dtype=float)
    positive = positive[positive > 0]
    if positive.size == 0:
        raise ValueError("No positive gaps to plot.")

    xmin = max(1e-3, float(positive.min()))
    xmax = min(float(positive.max()), float(xmax_sec))
    n_beyond = int((positive > xmax_sec).sum())

    # Heavy-tailed data: bin on log10(x) with equal-width bins so density is
    # per unit log10(sec) and the tail stays visible. The x-axis still reads
    # as decades — we just relabel ticks back to seconds.
    log_x = np.log10(positive)
    log_xmin, log_xmax = np.log10(xmin), np.log10(xmax)
    bins = np.linspace(log_xmin, log_xmax, n_bins)

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, ax = plt.subplots(figsize=(14, 5))

    counts, _, _ =ax.hist(log_x, bins=bins, color="#6C757D", edgecolor="white", linewidth=0.4, density=True)
    ax.set_xlabel("Inter-call gap (sec)")
    ax.set_ylabel("Density")
    ax.set_xlim(log_xmin, log_xmax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    decades = np.arange(int(np.ceil(log_xmin)), int(np.floor(log_xmax)) + 1)
    ax.set_xticks(decades)
    ax.set_xticklabels([f"$10^{{{d}}}$" for d in decades])
    ax.set_yticks([0, round(counts.max(),2)])
    ax.set_yticklabels([0, round(counts.max(),2)])
    ax.set_ylim(0, round(counts.max(),2))


    # for thresh in thresholds:
    #     ax.axvline(np.log10(thresh), color="#E63946", linestyle="--", linewidth=1.0, alpha=0.7)

    full_title = title
    if n_beyond:
        full_title = f"{full_title}  —  {n_beyond:,} gaps > {xmax_sec:g}s not shown"
    full_title = ""
    ax.set_title(full_title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {pdf_path}")


def main() -> int:
    if not DATE_FOLDERS:
        raise SystemExit("Set DATE_FOLDERS to at least one date.")
    dates_tag = "_".join(DATE_FOLDERS)
    out_name = OUTPUT_NAME or f"icg_histogram_{dates_tag}.png"
    out_path = OUTPUT_BASE / "icg_histograms" / out_name

    print(f"Loading parquet for: {DATE_FOLDERS}")
    df = load_calls(DATE_FOLDERS)
    print(f"  {len(df):,} calls loaded; computing per-experiment gaps...")
    gaps = gaps_per_experiment(df)
    print(f"  {gaps.size:,} positive gaps")

    title = f"Inter-call gap ({dates_tag}, n={gaps.size:,})"

    plot_histogram(gaps, THRESHOLDS, N_BINS, XMAX_SEC, title, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
