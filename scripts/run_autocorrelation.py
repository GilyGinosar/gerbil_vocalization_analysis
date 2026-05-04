"""Autocorrelation of gerbil vocalization time series, chunked in real time.

Pools experiments within each date folder (cohort) by real clock time, then
slices each channel's record into non-overlapping chunks of CHUNK_HOURS.
Each (channel, chunk) becomes one independent observation: a binary series
x(t) at dt=0.1s (1 = at least one call started in that bin, 0 = silent),
demeaned by that chunk's mean, autocorrelation C(tau) via FFT (Wiener-Khinchin)
with the unbiased 1/(N-tau) normalization. Within a cohort, the mean curve
is the unweighted mean across chunks (all chunks have the same duration) and
the SE is the across-chunks std / sqrt(n_chunks). This makes the SE band
sensitive to *day-to-day variability* rather than channel-to-channel
variability, which is what you want when looking for circadian peaks.

Lag axis: a chunk of W hours with MAX_LAG_FRACTION=0.5 reaches tau = W/2 hours,
i.e. CHUNK_HOURS=48 reaches 24 h (the circadian peak sits at the noisy edge);
CHUNK_HOURS=96 reaches 48 h (24 h with margin, can also see 48 h repetition).

Outputs, per date folder:
  - autocorrelation_<date>_aggregate.png       (all calls pooled)
  - autocorrelation_<date>_by_call_type.png    (one line per event_type)

Both plots overlay a reference power-law line of slope -0.36 (corresponds to
Delta=0.18 from Bialek & Shaevitz 2024 fly-behavior paper). It is a visual
reference, not a fit.

Edit the values below and run:  python scripts/run_autocorrelation.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === Edit these before running ============================================
DATE_FOLDERS = ["2025_03","2025_07", "2025_10", "2026_02"]

CHUNK_HOURS         = 96      # chunk length per (channel, chunk) observation
DT_SEC              = 0.1     # bin width
MAX_LAG_FRACTION    = 0.5     # cap lags at chunk_length * this
MIN_CALLS_PER_CHUNK = 50      # drop a chunk below this count (low SNR)
MIN_CALLS_PER_TYPE  = 500     # skip per-type analysis when below this in cohort
MIN_CHUNKS          = 3       # need at least this many usable chunks to plot

# Reference power-law slope on log-log: C(tau) ~ tau^slope.
# Delta=0.18 from the fly paper -> slope = -2*Delta = -0.36.
REFERENCE_SLOPE = -0.36

# Number of log-spaced lag points to plot per curve (subsample for clean plots).
N_LOG_LAGS = 200

PARQUET_DIR = Path(
    "/Users/gilyginosar/Dropbox (Personal)/Vocalizations_project/Data/parquet_cache"
)
OUTPUT_DIR  = Path(
    "/Users/gilyginosar/Dropbox (Personal)/Vocalizations_project/Data/autocorrelation_outputs"
)
# ==========================================================================


def autocorr_fft_unbiased(y: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation via FFT.

    Returns ac[tau] for tau = 0..N-1, where
        ac[tau] = (1/(N - tau)) * sum_{t=0}^{N-1-tau} y[t] * y[t+tau].
    """
    n = y.size
    nfft = 1 << ((2 * n - 1).bit_length())  # next power of two >= 2n - 1
    F = np.fft.rfft(y, n=nfft)
    raw = np.fft.irfft(F * F.conj(), n=nfft)[:n]
    norm = np.arange(n, 0, -1, dtype=np.float64)  # N, N-1, ..., 1
    return raw / norm


def channel_autocorrelation(call_starts_sec: np.ndarray, T_sec: float,
                             dt: float, max_lag_frac: float):
    """Return (ac_normalized, n_bins) for one channel, or (None, n_bins)
    if the demeaned series has no variance."""
    n_bins = int(np.ceil(T_sec / dt))
    if n_bins <= 1:
        return None, n_bins
    x = np.zeros(n_bins, dtype=np.float64)
    idx = np.floor(call_starts_sec / dt).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    if idx.size == 0:
        return None, n_bins
    x[idx] = 1.0  # binary indicator: at least one call started in this bin
    y = x - x.mean()
    ac = autocorr_fft_unbiased(y)
    if ac[0] <= 0:
        return None, n_bins
    ac /= ac[0]  # so C(0) = 1
    max_lag = max(1, int(n_bins * max_lag_frac))
    return ac[:max_lag], n_bins


def iter_chunks(df: pd.DataFrame, chunk_hours: float):
    """Yield non-overlapping fixed-length chunks per channel in real clock time.

    Each yielded dict has keys:
      channel, chunk_id, t0 (chunk start as Timestamp), duration_sec, rows
    where `rows` is the slice of `df` whose start_time_real falls in [t0, t0+W).
    The chunk grid is anchored at the channel's earliest start_time_real, and
    only full chunks are emitted (the trailing partial chunk is dropped so
    every observation has the same N).
    """
    chunk_td = pd.Timedelta(hours=chunk_hours)
    starts_all = pd.to_datetime(df["start_time_real"])
    stops_all  = pd.to_datetime(df["stop_time_real"])

    for chan, sub in df.groupby("channel"):
        sub_starts = starts_all.loc[sub.index]
        ref = sub_starts.min()
        end = stops_all.loc[sub.index].max()
        n_full = int(np.floor((end - ref) / chunk_td))
        for i in range(n_full):
            t0 = ref + i * chunk_td
            t1 = t0 + chunk_td
            mask = (sub_starts >= t0) & (sub_starts < t1)
            yield {
                "channel": int(chan),
                "chunk_id": i,
                "t0": t0,
                "duration_sec": chunk_td.total_seconds(),
                "rows": sub.loc[sub.index[mask.to_numpy()]],
            }


def per_chunk_acs(df: pd.DataFrame, dt: float, max_lag_frac: float,
                   chunk_hours: float, call_type: str | None,
                   min_calls_per_chunk: int):
    """Compute one autocorrelation per (channel, chunk).

    Returns (acs, labels, n_calls_per_chunk). All chunks share the same
    duration so weights are uniform; the caller can treat the list as iid
    observations and take a plain mean / SE.
    """
    acs: list[np.ndarray] = []
    labels: list[tuple[int, int]] = []
    n_calls_list: list[int] = []

    for chunk in iter_chunks(df, chunk_hours):
        rows = chunk["rows"]
        if call_type is not None:
            rows = rows[rows["event_type"] == call_type]
        n = len(rows)
        if n < min_calls_per_chunk:
            continue
        starts_sec = (
            pd.to_datetime(rows["start_time_real"]) - chunk["t0"]
        ).dt.total_seconds().to_numpy()
        ac, _ = channel_autocorrelation(
            starts_sec, chunk["duration_sec"], dt, max_lag_frac
        )
        if ac is None:
            continue
        acs.append(ac)
        labels.append((chunk["channel"], chunk["chunk_id"]))
        n_calls_list.append(n)

    return acs, labels, n_calls_list


def aggregate(acs):
    """Stack ACs (all the same length, since chunks are fixed duration) and
    return (mean, se). SE = std-across-chunks / sqrt(n_chunks) at each lag."""
    if not acs:
        return None, None
    A = np.stack(acs, axis=0)  # (n_chunks, n_lags)
    mean = np.nanmean(A, axis=0)
    n = A.shape[0]
    if n > 1:
        sd = np.nanstd(A, axis=0, ddof=1)
        se = sd / np.sqrt(n)
    else:
        se = np.zeros_like(mean)
    return mean, se


def log_indices(max_lag: int, n_points: int) -> np.ndarray:
    """Unique log-spaced integer lag indices in [1, max_lag-1]."""
    if max_lag <= 1:
        return np.array([], dtype=int)
    return np.unique(
        np.round(np.geomspace(1, max_lag - 1, min(n_points, max_lag - 1)))
    ).astype(int)


def plot_curve(ax, mean, se, dt, label, color, n_log_lags=N_LOG_LAGS):
    if mean is None or mean.size <= 1:
        return
    idx = log_indices(mean.size, n_log_lags)
    if idx.size == 0:
        return
    lags = idx * dt
    m = mean[idx]
    s = se[idx] if se is not None else np.zeros_like(m)
    keep = np.isfinite(m) & (m > 0)
    if not keep.any():
        return
    ax.plot(lags[keep], m[keep], lw=1.8, color=color, label=label)
    lower = np.maximum(m[keep] - s[keep], 1e-10)  # clip for log scale
    upper = m[keep] + s[keep]
    ax.fill_between(lags[keep], lower, upper, color=color, alpha=0.2, linewidth=0)


def add_reference_powerlaw(ax, slope, dt, max_lag_sec, anchor=(1.0, 0.1)):
    """Dashed power-law guide line on log-log axes."""
    x0, y0 = anchor
    x = np.array([dt, max(max_lag_sec, dt * 10)])
    y = y0 * (x / x0) ** slope
    ax.plot(x, y, color="gray", ls="--", lw=1.2,
            label=f"power-law slope {slope:g}")


def style_axes(ax, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lag tau (s)")
    ax.set_ylabel("normalized autocorrelation C(tau)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def add_circadian_guides(ax, max_lag_sec):
    """Vertical dotted lines at 24, 48, 72 h to spot circadian peaks."""
    for hours in (24, 48, 72):
        tau = hours * 3600.0
        if tau <= max_lag_sec:
            ax.axvline(tau, color="black", ls=":", lw=0.8, alpha=0.6)
            ax.text(tau, ax.get_ylim()[1], f" {hours}h",
                    va="top", ha="left", fontsize=8, color="black", alpha=0.7)


def process_cohort(df: pd.DataFrame, date_folder: str, output_dir: Path):
    """Compute and plot autocorrelations for one cohort. Prints diagnostics."""

    # --- aggregate (all call types) ---
    acs, labels, n_calls_per_chunk = per_chunk_acs(
        df, DT_SEC, MAX_LAG_FRACTION, CHUNK_HOURS,
        call_type=None, min_calls_per_chunk=MIN_CALLS_PER_CHUNK,
    )
    n_chunks = len(acs)
    n_channels = len({chan for chan, _ in labels})
    print(
        f"[{date_folder}] n_chunks={n_chunks}  n_channels={n_channels}  "
        f"chunk_hours={CHUNK_HOURS}  "
        f"chunk_calls min={min(n_calls_per_chunk) if n_calls_per_chunk else 0} "
        f"median={int(np.median(n_calls_per_chunk)) if n_calls_per_chunk else 0} "
        f"max={max(n_calls_per_chunk) if n_calls_per_chunk else 0}"
    )
    if n_chunks < MIN_CHUNKS:
        print(f"  [WARN] only {n_chunks} usable chunks; skipping cohort")
        return

    mean, se = aggregate(acs)
    max_lag_sec = mean.size * DT_SEC

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    plot_curve(ax, mean, se, DT_SEC,
               label=f"{date_folder} (n_chunks={n_chunks}, n_chan={n_channels})",
               color="C0")
    add_reference_powerlaw(ax, REFERENCE_SLOPE, DT_SEC, max_lag_sec)
    style_axes(ax, f"{date_folder}  |  aggregate (all call types), {CHUNK_HOURS}h chunks")
    add_circadian_guides(ax, max_lag_sec)
    ax.legend()
    fig.tight_layout()
    out_agg = output_dir / f"autocorrelation_{date_folder}_aggregate.png"
    fig.savefig(out_agg, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_agg}")

    # --- per call type ---
    type_counts = df["event_type"].value_counts()
    per_type = {}
    for ct, n_calls in type_counts.items():
        if n_calls < MIN_CALLS_PER_TYPE:
            print(
                f"  [WARN] skipping call type '{ct}' "
                f"(n_calls={n_calls} < {MIN_CALLS_PER_TYPE})"
            )
            continue
        ac_t, lbl_t, _ = per_chunk_acs(
            df, DT_SEC, MAX_LAG_FRACTION, CHUNK_HOURS,
            call_type=ct, min_calls_per_chunk=MIN_CALLS_PER_CHUNK,
        )
        if len(ac_t) < MIN_CHUNKS:
            print(
                f"  [WARN] call type '{ct}' has only {len(ac_t)} chunks "
                f"(>= {MIN_CALLS_PER_CHUNK} calls each); skipping"
            )
            continue
        m_t, se_t = aggregate(ac_t)
        per_type[ct] = (m_t, se_t, len(ac_t))

    if per_type:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        cmap = plt.get_cmap("tab10")
        max_lag_sec = max(m.size for (m, _, _) in per_type.values()) * DT_SEC
        for i, (ct, (m_t, se_t, n_ch_t)) in enumerate(per_type.items()):
            plot_curve(ax, m_t, se_t, DT_SEC,
                       label=f"{ct} (n_chunks={n_ch_t}, n_calls={int(type_counts[ct])})",
                       color=cmap(i))
        add_reference_powerlaw(ax, REFERENCE_SLOPE, DT_SEC, max_lag_sec)
        style_axes(ax, f"{date_folder}  |  per call type, {CHUNK_HOURS}h chunks")
        add_circadian_guides(ax, max_lag_sec)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_pt = output_dir / f"autocorrelation_{date_folder}_by_call_type.png"
        fig.savefig(out_pt, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_pt}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    for date_folder in DATE_FOLDERS:
        path = PARQUET_DIR / f"all_calls_{date_folder}.parquet"
        if not path.exists():
            print(f"[{date_folder}] missing parquet at {path}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        process_cohort(df, date_folder, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())