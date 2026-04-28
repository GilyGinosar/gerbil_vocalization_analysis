"""Compute call rate (calls / minute / gerbil) per experiment, summarised per date folder.

Pulls all calls together (arena 1 + arena 2 + underground -- no channel filter),
divides by the real recording duration of that experiment (summed chunk durations
from sync.csv, so gaps within an experiment shorten the denominator and missing
experiments don't dilute their date folder), divides by NUM_GERBILS.

Pipeline:
  1. For each date folder in DATE_FOLDERS, list integer-named experiment subfolders.
  2. For each experiment:
       num_calls    = len(calls.csv) (and per call type via event_type)
       duration_min = sum_chunks(end - start) from sync.csv, in minutes
       rate         = num_calls / duration_min / NUM_GERBILS
     Call types are canonicalised via calc_transitions._canonicalize_call_type.
     Types present in the global universe but missing from a given experiment
     contribute a 0 rate for that experiment (so the mean isn't biased upward
     by averaging only over experiments that happened to detect that type).
  3. Per date folder: mean +/- std (sample, ddof=1) of per-experiment rates,
     pooled and per call type.
  4. Write four CSVs to <BASE_PROCESSED>/<OUTPUT_FOLDER_NAME>/<dates_tag>/:
       per_experiment_rates.csv          (pooled, one row per exp)
       summary.csv                       (pooled, one row per date folder)
       per_experiment_rates_by_type.csv  (long: one row per exp x call_type)
       summary_by_type.csv               (long: one row per date_folder x call_type)

Edit the values below and run:  python scripts/run_call_rate.py
"""
import ast
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.audio_processing_config import get_experiment_month
from vocalization_analysis.calc_transitions import _canonicalize_call_type


# === Edit these before running ============================================
DATE_FOLDERS = ["2025_07", "2025_10", "2026_02"]

# All channels are pooled (arena 1 + arena 2 + underground). Number of animals
# the rate is normalised by.
NUM_GERBILS = 6

OUTPUT_FOLDER_NAME = "call_rate_outputs"
# ==========================================================================


if platform.system() == "Windows":
    BASE_RAW = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data")
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    BASE_RAW = Path("/mnt/home/neurostatslab/ceph/saneslab_data/big_setup/")
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def get_experiment_audio_dir(exp: int) -> Path:
    return BASE_PROCESSED / get_experiment_month(exp) / str(exp)


def list_experiment_ids_for_date(date_folder: str) -> list[int]:
    folder = BASE_PROCESSED / date_folder
    if not folder.exists():
        raise SystemExit(f"Date folder not found: {folder}")
    return sorted(
        int(p.name) for p in folder.iterdir()
        if p.is_dir() and p.name.isdigit()
    )


def get_experiment_sync_path(exp: int) -> Path:
    return BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync" / "sync.csv"


def _parse_sync_field(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def experiment_duration_minutes(exp: int) -> float:
    """Real recording duration in minutes: sum of (chunk_end - chunk_start) across
    all sync.csv rows. Robust to within-experiment gaps."""
    sync_path = get_experiment_sync_path(exp)
    if not sync_path.exists():
        raise FileNotFoundError(f"sync.csv not found: {sync_path}")
    sync_df = pd.read_csv(sync_path)
    if "timestamp" not in sync_df.columns:
        raise ValueError(f"sync.csv missing 'timestamp' column: {sync_path}")

    timestamps = sync_df["timestamp"].apply(_parse_sync_field)
    starts = pd.to_datetime(timestamps.apply(lambda t: t[0]))
    stops  = pd.to_datetime(timestamps.apply(lambda t: t[1]))
    chunk_sec = (stops - starts).dt.total_seconds()
    chunk_sec = chunk_sec[chunk_sec > 0]
    if chunk_sec.empty:
        raise ValueError(f"sync.csv for exp {exp} has no positive-duration chunks")
    return float(chunk_sec.sum()) / 60.0


def load_call_counts(exp: int) -> tuple[int, dict[str, int]]:
    """Return (total_calls, calls_per_canonicalised_type) for `exp`."""
    calls_path = get_experiment_audio_dir(exp) / "calls.csv"
    if not calls_path.exists():
        raise FileNotFoundError(f"calls.csv not found: {calls_path}")
    df = pd.read_csv(calls_path)
    total = len(df)
    if "event_type" not in df.columns:
        raise ValueError(f"calls.csv missing 'event_type' column: {calls_path}")
    if total == 0:
        return 0, {}
    types = df["event_type"].map(_canonicalize_call_type)
    by_type = types.value_counts().to_dict()
    return total, {str(k): int(v) for k, v in by_type.items()}


def plot_pooled_rates(per_exp: pd.DataFrame, summary: pd.DataFrame,
                      date_folders: list[str], output_path: Path) -> None:
    """Bar = mean rate per date folder, error bar = std, dots = per-experiment rates."""
    x = np.arange(len(date_folders))
    means = summary.set_index("date_folder").reindex(date_folders)["mean_rate"].to_numpy()
    stds  = summary.set_index("date_folder").reindex(date_folders)["std_rate"].to_numpy()
    stds_for_plot = np.where(np.isnan(stds), 0.0, stds)

    fig, ax = plt.subplots(figsize=(1.6 * len(date_folders) + 2, 4.5))
    ax.bar(x, means, yerr=stds_for_plot, capsize=6, color="#457B9D", alpha=0.7,
           edgecolor="black", linewidth=0.8, zorder=2)

    rng = np.random.default_rng(0)
    for xi, d in zip(x, date_folders):
        rates = per_exp.loc[per_exp["date_folder"] == d, "rate_calls_per_min_per_gerbil"].to_numpy()
        if len(rates) == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(rates))
        ax.scatter(np.full_like(rates, xi, dtype=float) + jitter, rates,
                   color="black", s=22, zorder=3, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(date_folders)
    ax.set_ylabel("calls / min / gerbil")
    ax.set_title("Pooled call rate per date folder (mean ± std, dots = experiments)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_by_type_rates(summary_by_type: pd.DataFrame,
                       date_folders: list[str], output_path: Path) -> None:
    """Grouped bars: x = call type, groups = date folder. Bar = mean, error bar = std.
    Call types ordered by descending overall (across-folders) mean rate."""
    type_order = (
        summary_by_type.groupby("call_type")["mean_rate"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    n_types = len(type_order)
    n_groups = len(date_folders)
    if n_types == 0:
        return

    x = np.arange(n_types)
    bar_w = 0.8 / n_groups
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(7, 1.1 * n_types + 2), 4.8))
    pivot_mean = summary_by_type.pivot(index="call_type", columns="date_folder", values="mean_rate").reindex(type_order)
    pivot_std  = summary_by_type.pivot(index="call_type", columns="date_folder", values="std_rate").reindex(type_order)
    for gi, d in enumerate(date_folders):
        if d not in pivot_mean.columns:
            continue
        means = pivot_mean[d].to_numpy()
        stds  = pivot_std[d].to_numpy()
        stds_for_plot = np.where(np.isnan(stds), 0.0, stds)
        offset = (gi - (n_groups - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds_for_plot, capsize=3,
               label=d, color=cmap(gi % 10), edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(type_order, rotation=30, ha="right")
    ax.set_ylabel("calls / min / gerbil")
    ax.set_title("Call rate by type (mean ± std across experiments)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="date folder", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    if not DATE_FOLDERS:
        raise SystemExit("Set DATE_FOLDERS to at least one date folder.")
    if NUM_GERBILS <= 0:
        raise SystemExit("NUM_GERBILS must be > 0.")

    dates_tag = "_".join(DATE_FOLDERS)
    output_dir = BASE_PROCESSED / OUTPUT_FOLDER_NAME / dates_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Date folders : {DATE_FOLDERS}")
    print(f"Num gerbils  : {NUM_GERBILS}")
    print(f"Output dir   : {output_dir}")
    print()

    rows: list[dict] = []
    type_rows: list[dict] = []  # one row per (exp, call_type) seen in that exp
    failed: list[tuple[int, str, str]] = []  # (exp, date_folder, reason)

    for date_folder in DATE_FOLDERS:
        exps = list_experiment_ids_for_date(date_folder)
        print(f"[{date_folder}] {len(exps)} experiments")
        for exp in exps:
            try:
                duration_min = experiment_duration_minutes(exp)
                num_calls, by_type = load_call_counts(exp)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  exp {exp}: skipped ({exc})")
                failed.append((exp, date_folder, str(exc)))
                continue

            rate = num_calls / duration_min / NUM_GERBILS
            rows.append({
                "date_folder": date_folder,
                "exp": exp,
                "num_calls": num_calls,
                "duration_min": duration_min,
                "rate_calls_per_min_per_gerbil": rate,
            })
            for call_type, n in by_type.items():
                type_rows.append({
                    "date_folder": date_folder,
                    "exp": exp,
                    "call_type": call_type,
                    "num_calls": n,
                    "duration_min": duration_min,
                    "rate_calls_per_min_per_gerbil": n / duration_min / NUM_GERBILS,
                })

    if not rows:
        raise SystemExit("No experiments produced a rate; nothing to summarise.")

    per_exp = pd.DataFrame(rows).sort_values(["date_folder", "exp"]).reset_index(drop=True)
    per_exp_path = output_dir / "per_experiment_rates.csv"
    per_exp.to_csv(per_exp_path, index=False)

    summary = (
        per_exp.groupby("date_folder")
        .agg(
            n_experiments=("exp", "size"),
            total_calls=("num_calls", "sum"),
            total_duration_min=("duration_min", "sum"),
            mean_rate=("rate_calls_per_min_per_gerbil", "mean"),
            std_rate=("rate_calls_per_min_per_gerbil", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
        )
        .reindex(DATE_FOLDERS)  # preserve config order
        .reset_index()
    )
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # Per-call-type breakdown.
    # Universe of call types = everything seen across all loaded experiments.
    # Backfill 0s so each loaded experiment has a row for every type, otherwise
    # types that only appear in some exps would get an inflated mean.
    all_types = sorted({r["call_type"] for r in type_rows})
    sparse = pd.DataFrame(type_rows)
    exp_index = per_exp[["date_folder", "exp", "duration_min"]].drop_duplicates()
    grid = exp_index.merge(pd.DataFrame({"call_type": all_types}), how="cross")
    by_type = grid.merge(
        sparse[["date_folder", "exp", "call_type", "num_calls"]],
        on=["date_folder", "exp", "call_type"],
        how="left",
    )
    by_type["num_calls"] = by_type["num_calls"].fillna(0).astype(int)
    by_type["rate_calls_per_min_per_gerbil"] = (
        by_type["num_calls"] / by_type["duration_min"] / NUM_GERBILS
    )
    by_type = by_type.sort_values(["date_folder", "call_type", "exp"]).reset_index(drop=True)
    by_type_path = output_dir / "per_experiment_rates_by_type.csv"
    by_type.to_csv(by_type_path, index=False)

    summary_by_type = (
        by_type.groupby(["date_folder", "call_type"])
        .agg(
            n_experiments=("exp", "size"),
            total_calls=("num_calls", "sum"),
            total_duration_min=("duration_min", "sum"),
            mean_rate=("rate_calls_per_min_per_gerbil", "mean"),
            std_rate=("rate_calls_per_min_per_gerbil", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")),
        )
        .reset_index()
    )
    # Preserve config order for date_folder.
    summary_by_type["date_folder"] = pd.Categorical(
        summary_by_type["date_folder"], categories=DATE_FOLDERS, ordered=True
    )
    summary_by_type = summary_by_type.sort_values(["date_folder", "call_type"]).reset_index(drop=True)
    summary_by_type_path = output_dir / "summary_by_type.csv"
    summary_by_type.to_csv(summary_by_type_path, index=False)

    # Wide pivot for at-a-glance reading: rows = call_type, cols = "<date>_mean" / "<date>_std".
    mean_pivot = summary_by_type.pivot(index="call_type", columns="date_folder", values="mean_rate")
    std_pivot  = summary_by_type.pivot(index="call_type", columns="date_folder", values="std_rate")
    mean_pivot.columns = [f"{c}_mean" for c in mean_pivot.columns]
    std_pivot.columns  = [f"{c}_std"  for c in std_pivot.columns]
    wide = pd.concat([mean_pivot, std_pivot], axis=1)
    interleaved_cols: list[str] = []
    for d in DATE_FOLDERS:
        interleaved_cols += [f"{d}_mean", f"{d}_std"]
    wide = wide.reindex(columns=[c for c in interleaved_cols if c in wide.columns])

    print()
    print("Per-experiment rates (pooled):")
    print(per_exp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Summary, pooled (calls / min / gerbil):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Summary by call type (calls / min / gerbil), mean +/- std across experiments:")
    print(wide.to_string(float_format=lambda x: f"{x:.4f}"))
    pooled_plot_path = output_dir / "call_rate_pooled.png"
    plot_pooled_rates(per_exp, summary, DATE_FOLDERS, pooled_plot_path)
    by_type_plot_path = output_dir / "call_rate_by_type.png"
    plot_by_type_rates(summary_by_type, DATE_FOLDERS, by_type_plot_path)

    print()
    print(f"Wrote: {per_exp_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {by_type_path}")
    print(f"Wrote: {summary_by_type_path}")
    print(f"Wrote: {pooled_plot_path}")
    print(f"Wrote: {by_type_plot_path}")

    if failed:
        print("\nSkipped experiments:")
        for exp, date_folder, reason in failed:
            print(f"  [{date_folder}] {exp}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
