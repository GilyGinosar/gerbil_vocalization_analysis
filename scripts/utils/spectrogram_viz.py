"""Audio reading + spectrogram/segment plotting helpers.

Copied verbatim from das_pipeline (audio.py, plotting.py) so that the
complex-call hand-labeling analysis can live in this repo without depending on
das_pipeline. Only the self-contained pieces the analysis actually uses were
brought over: read_audio, plot_spectrogram, plot_segments_overlay, COLOR_MAP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from scipy.io import wavfile
from scipy.signal import spectrogram


COLOR_MAP: dict[str, str] = {
    "alarm": "tab:red",
    "high-freq": "tab:orange",
    "newborn": "tab:green",
    "stacks": "tab:blue",
    "warble": "tab:purple",
    "noise": "0.75",
}


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(path)
    if x.ndim == 1:
        x = np.atleast_2d(x).T
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32, copy=False)
    return fs, x


def plot_spectrogram(
    ax: Axes,
    x: np.ndarray,
    fs: int,
    *,
    channel: int = 0,
    nperseg: int = 1024,
    noverlap: int = 768,
    min_freq: float = 1000.0,
    max_freq: float = 60000.0,
    vmin: float = -40.0,
    vmax: float = 0.0,
    cmap: str = "viridis",
    t_start: float = 0.0,
) -> QuadMesh:
    if x.ndim == 2:
        x = x[:, channel]
    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    Sxx_db = 20.0 * np.log10(Sxx + 1e-12)
    Sxx_db = Sxx_db - np.max(Sxx_db)
    freq_mask = (f >= min_freq) & (f <= max_freq)
    mesh = ax.pcolormesh(
        t + t_start,
        f[freq_mask] / 1000.0,
        Sxx_db[freq_mask, :],
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_ylim(min_freq / 1000.0, max_freq / 1000.0)
    return mesh


def plot_segments_overlay(
    ax: Axes,
    segments_df: pd.DataFrame,
    *,
    color_map: dict[str, str] | None = None,
    default_color: str = "tab:gray",
    alpha_span: float = 0.10,
    alpha_line: float = 0.55,
    with_text: bool = True,
    text_y_frac: float = 0.96,
    text_fontsize: int = 8,
) -> None:
    cmap = color_map if color_map is not None else COLOR_MAP
    y_min, y_max = ax.get_ylim()
    y_text = y_min + (y_max - y_min) * text_y_frac
    for _, row in segments_df.iterrows():
        label = str(row["label"])
        onset = float(row["onset_s"])
        offset = float(row["offset_s"])
        color = cmap.get(label, default_color)
        ax.axvspan(onset, offset, color=color, alpha=alpha_span)
        ax.axvline(onset, color=color, linewidth=0.8, alpha=alpha_line)
        ax.axvline(offset, color=color, linewidth=0.8, alpha=alpha_line)
        if with_text:
            ax.text(onset, y_text, label, color=color, fontsize=text_fontsize, va="top")
