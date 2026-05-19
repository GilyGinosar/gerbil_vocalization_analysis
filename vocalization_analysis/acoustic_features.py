"""Per-call acoustic feature extraction, via vocalpy.

A thin batch-friendly wrapper around two vocalpy feature families:
    - SAT (Sound Analysis Toolbox-style, vocalpy.feature.sat.similarity_features):
      time-resolved contours of pitch / amplitude / entropy / FM / goodness-of-pitch,
      reduced here to per-call scalars (medians, max, min, peak-at-loudest-frame, ...).
    - Soundsig spectral envelope (vocalpy.feature.soundsig.features.spectral_envelope_features):
      PSD-shape scalars (mean / std / skew / kurtosis / entropy / quartiles).

Defaults are tuned for **gerbil USVs** (sr=125 kHz, Nyquist=62.5 kHz). We bypass
the friendlier `predefined_acoustic_features` wrapper and call
`spectral_envelope_features` directly so we can override its hard-coded
`f_high=10 kHz` default - that default would discard nearly all gerbil
USV signal energy.

Path helpers (`call_wav_path`, `load_call_slice`, `load_call_with_context`) take
`base_audio` as an explicit first argument so this module stays platform-pure
(no cluster-vs-Mac path knowledge).

Public entry points:
    SAT_PARAMS              : dict of SAT params for similarity_features
    FEATURE_COLUMNS         : list of all scalar feature names returned
    compute_features        : run vocalpy on (y, sr) -> dict of 20 scalars
    call_wav_path           : locate the WAV file for one call
    load_call_slice         : read just the syllable's samples
    load_call_with_context  : read a 1-second window around the call
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import soundfile as sf

from vocalpy import Sound
from vocalpy.feature.sat import similarity_features
from vocalpy.feature.soundsig.features import spectral_envelope_features


# --------------------------------------------------------------------------
# Parameters - tuned for gerbil USVs (sr=125 kHz, Nyquist=62.5 kHz).
# --------------------------------------------------------------------------
SAT_PARAMS = dict(
    n_fft            = 512,      # ~4 ms per frame at sr=125 kHz
    hop_length       = 256,      # ~2 ms hop
    min_freq         = 1_000,    # ignore energy below 1 kHz
    max_F0           = 62_500,   # pitch ceiling (near Nyquist)
    fmax_yin         = 62_500,
    trough_threshold = 0.5,
)

# Soundsig PSD parameters. f_high=60 kHz is critical (default 10 kHz would
# throw away almost all gerbil USV energy).
SOUNDSIG_F_HIGH    = 60_000
SOUNDSIG_NFFT      = 512
SOUNDSIG_NOVERLAP  = 256


# --------------------------------------------------------------------------
# Per-call scalars returned by compute_features. NaN if extraction fails.
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # SAT-derived scalars
    "duration_s",

    # Pitch (YIN) reductions
    "peak_freq_hz",                  # pitch at the loudest frame
    "start_pitch_hz",                # median pitch in first 1/3 of frames
    "stop_pitch_hz",                 # median pitch in last  1/3 of frames
    "pitch_median_hz",
    "max_pitch_hz",                  # nan-max of YIN pitch contour (catches upsweeps)
    "min_pitch_hz",
    "pitch_slope_hz_per_s",          # (stop - start) / duration; linear pitch slope
    "bandwidth_hz",                  # |stop - start|
    "pitch_curvature_hz_per_call2",  # quadratic coef of pitch vs normalized time [0,1];
                                     # negative = inverted-U (rises then falls)

    # Other SAT contour reductions
    "entropy_median",                # Wiener entropy: 0=pure tone, 1=noise
    "fm_median",                     # vocalpy FM contour, median (radians, [0, pi/2])
    "fm_early",                      # median FM in first 1/3 of frames
    "fm_late",                       # median FM in last  1/3 of frames
    "fm_slope_per_s",                # linear slope of FM contour, rad/s
    "goodness_of_pitch_median",      # how harmonic / sinusoidal

    # Soundsig spectral envelope (PSD-shape stats over the full call)
    "mean_s_hz",                     # center-of-mass of the PSD
    "std_s_hz",                      # spread of the PSD
    "skew_s",                        # PSD asymmetry (dimensionless)
    "kurtosis_s",                    # PSD peakiness
    "entropy_s",                     # PSD entropy normalized to log2(n_bins)
    "q1_hz",                         # 25th percentile frequency of total power
    "q2_hz",                         # 50th percentile (median frequency of power)
    "q3_hz",                         # 75th percentile
]
NAN_RESULT = {col: np.nan for col in FEATURE_COLUMNS}


# --------------------------------------------------------------------------
# File-path / audio-loading helpers.
# --------------------------------------------------------------------------

def call_wav_path(base_audio, date_folder, exp, channel, file_num):
    """Path to the concatenated channel WAV that contains a given call.

    The caller passes `base_audio` (project-specific top-level audio dir) so
    this module doesn't need to know about Mac vs cluster paths.
    """
    return (
        Path(base_audio)
        / date_folder
        / str(int(exp))
        / "Averaged_wavs_w_annotations"
        / f"channel_{int(channel)}_file_{int(file_num):03d}.wav"
    )


@functools.lru_cache(maxsize=256)
def _wav_metadata(path_str):
    """Return (samplerate_hz, duration_seconds) for the file. Cached."""
    info = sf.info(path_str)
    return info.samplerate, info.frames / info.samplerate


def load_call_slice(base_audio, date_folder, exp, channel, file_num,
                    start_sec, stop_sec, pad_sec=0.0):
    """Read just the syllable's samples out of its concatenated channel WAV.

    Parameters
    ----------
    pad_sec : float, default 0.0
        Seconds of audio to include on each side of [start_sec, stop_sec].
        Useful when the segmenter sometimes clips a call's tail - a small
        pad (e.g. 0.01) lets feature extraction see the full call.
        The actual read is clamped to [0, file_duration].
        Note: if you pad, `len(y) / sr` overstates the call's duration.
        Override `feats["duration_s"]` downstream with the segmenter's
        original (stop_sec - start_sec) if that matters.

    Returns
    -------
    y  : 1D float32 numpy array
    sr : sample rate (Hz)
    """
    path = call_wav_path(base_audio, date_folder, exp, channel, file_num)
    sr, file_dur = _wav_metadata(str(path))

    read_start = max(0.0, start_sec - pad_sec)
    read_stop  = min(file_dur, stop_sec + pad_sec)

    y, _ = sf.read(
        str(path),
        start=int(round(read_start * sr)),
        stop=int(round(read_stop  * sr)),
        dtype="float32", always_2d=False,
    )
    return y, sr


def load_call_with_context(base_audio, date_folder, exp, channel, file_num,
                           start_sec, stop_sec, window_sec=1.0):
    """Read `window_sec` of audio centered on a call (clamped to file edges).

    Returns
    -------
    y_window : 1D float32 array
    sr       : sample rate (Hz)
    t0_call  : seconds within y_window where the call starts
    t1_call  : seconds within y_window where the call stops
    """
    path = call_wav_path(base_audio, date_folder, exp, channel, file_num)
    sr, file_dur = _wav_metadata(str(path))
    call_dur = stop_sec - start_sec

    # Center the call: pad equally on both sides, clamp to file boundaries.
    half_pad  = max(0.0, (window_sec - call_dur) / 2)
    win_start = max(0.0, start_sec - half_pad)
    win_stop  = min(file_dur, win_start + max(window_sec, call_dur))
    win_start = max(0.0, win_stop - max(window_sec, call_dur))

    y_window, _ = sf.read(
        str(path),
        start=int(round(win_start * sr)),
        stop =int(round(win_stop  * sr)),
        dtype="float32", always_2d=False,
    )
    return y_window, sr, start_sec - win_start, stop_sec - win_start


# --------------------------------------------------------------------------
# The main feature extractor.
# --------------------------------------------------------------------------

def compute_features(y, sr):
    """Run vocalpy on one call's audio. Return a flat dict of scalars.

    Returns a dict with every key in `FEATURE_COLUMNS`. Values are floats
    where extraction succeeded, NaN where it didn't. We never raise so a
    batch loop can run over thousands of calls without crashing.
    """
    result = dict(NAN_RESULT)
    result["duration_s"] = len(y) / sr
    if len(y) < SAT_PARAMS["n_fft"]:
        return result

    # ----- SAT contours -> per-call scalars -----
    try:
        sound = Sound(data=y.astype(np.float32), samplerate=sr)
        sat_out = similarity_features(sound, **SAT_PARAMS)

        pitch     = np.asarray(sat_out.data.pitch.values).squeeze()
        amplitude = np.asarray(sat_out.data.amplitude.values).squeeze()
        entropy   = np.asarray(sat_out.data.entropy.values).squeeze()
        fm        = np.asarray(sat_out.data.frequency_modulation.values).squeeze()
        gop       = np.asarray(sat_out.data.goodness_of_pitch.values).squeeze()

        n_frames  = pitch.shape[-1]
        third_len = max(1, n_frames // 3)

        # --- Pitch (YIN) reductions ---
        result["start_pitch_hz"]  = float(np.nanmedian(pitch[:third_len]))
        result["stop_pitch_hz"]   = float(np.nanmedian(pitch[-third_len:]))
        result["pitch_median_hz"] = float(np.nanmedian(pitch))
        result["max_pitch_hz"]    = float(np.nanmax(pitch))
        result["min_pitch_hz"]    = float(np.nanmin(pitch))

        loudest_frame_idx = int(np.nanargmax(amplitude))
        result["peak_freq_hz"]    = float(pitch[loudest_frame_idx])

        # Linear PITCH slope (Hz/s): from start_pitch to stop_pitch.
        delta_hz = result["stop_pitch_hz"] - result["start_pitch_hz"]
        result["pitch_slope_hz_per_s"] = delta_hz / result["duration_s"]
        result["bandwidth_hz"]         = abs(delta_hz)

        # Pitch curvature: quadratic coefficient of pitch contour vs
        # NORMALIZED time in [0, 1]. Independent of call duration.
        # Sign: negative => inverted-U (rises then falls); positive => U.
        pitch_mask = ~np.isnan(pitch)
        if pitch_mask.sum() >= 5:
            t_norm = np.linspace(0, 1, len(pitch))[pitch_mask]
            a, _, _ = np.polyfit(t_norm, pitch[pitch_mask], 2)
            result["pitch_curvature_hz_per_call2"] = float(a)

        # --- Other SAT contour reductions ---
        result["entropy_median"]           = float(np.nanmedian(entropy))
        result["goodness_of_pitch_median"] = float(np.nanmedian(gop))

        # FM contour (radians, [0, pi/2]).
        result["fm_median"] = float(np.nanmedian(fm))
        result["fm_early"]  = float(np.nanmedian(fm[:third_len]))
        result["fm_late"]   = float(np.nanmedian(fm[-third_len:]))

        # Linear FM slope per SECOND (not per frame!) - one frame is
        # hop_length/sr seconds, so we divide the per-frame slope by frame_dt.
        fm_mask = ~np.isnan(fm)
        if fm_mask.sum() >= 5:
            frame_dt = SAT_PARAMS["hop_length"] / sr
            slope_per_frame, _ = np.polyfit(
                np.arange(len(fm))[fm_mask], fm[fm_mask], 1
            )
            result["fm_slope_per_s"] = float(slope_per_frame / frame_dt)
    except Exception:
        pass

    # ----- Soundsig spectral envelope -> PSD-shape scalars -----
    try:
        psd_stats = spectral_envelope_features(
            y.astype(np.float64),
            samplerate=sr,
            f_high=SOUNDSIG_F_HIGH,
            NFFT=SOUNDSIG_NFFT,
            noverlap=SOUNDSIG_NOVERLAP,
        )
        result["mean_s_hz"]  = float(psd_stats["mean_s"])
        result["std_s_hz"]   = float(psd_stats["std_s"])
        result["skew_s"]     = float(psd_stats["skew_s"])
        result["kurtosis_s"] = float(psd_stats["kurtosis_s"])
        result["entropy_s"]  = float(psd_stats["entropy_s"])
        result["q1_hz"]      = float(psd_stats["q1"])
        result["q2_hz"]      = float(psd_stats["q2"])
        result["q3_hz"]      = float(psd_stats["q3"])
    except Exception:
        pass

    return result
