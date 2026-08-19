# gerbil_vocalization_analysis

Detecting and analysing gerbil vocalizations from a multi-arena colony recording.
**Code lives here; data lives on ceph** (`.../Processed_data/Audio/`), never in the repo.

---

## The data chain

```
experiment_<id>/concatenated_data_cam_mic_sync/       6 raw mic channels + sync.csv
        │
        │  average mic pairs → 3 virtual channels (arena_1 / arena_2 / underground)
        ▼   scripts/pipeline/average_audio.py            [slurm/average-audio-array.sh]
Averaged_wavs_w_annotations/channel_{10,20,30}_file_NNN.wav
        │
        │  DAS (external tool) detects calls
        ▼
calls_confident/<params>/*_accepted_calls.csv
        │
        │  assign each call to an arena. TWO scripts do this, and both write the same file:
        │    combine_exp_calls.py    channel → arena directly  ← used for 2025_07 / 2025_10 / 2026_02
        │    run_rms_assignment.py   RMS across channels + overlap dedupe  (needs WAV reads)
        ▼
<exp>/calls.csv                                        one row per call, with location
        │
        │  pool onto one real-time axis, write to ceph, cache as parquet — one command
        ▼   scripts/pipeline/pool_calls.py --date-folder 2026_08
all_calls/all_calls_<date>.csv
all_calls/parquet_cache/all_calls_<date>.parquet       ← what every analysis script reads
```

Video runs alongside, and lands on the same clock:

```
Video/<date>/<exp>/video_<camera>_<file_num>.csv    per-frame detections (tracking repo)
        │
        │  attach exp / location / start_time_real, via the same sync.csv
        ▼   scripts/pipeline/pool_detections.py --date-folder 2026_02
Video/<date>/<exp>/detections.parquet   + files_vetted.csv   per experiment, EVERY row
        │
        │  drop detections the detector locked onto a fixed object
        │  (`stationary`; ~7% of 2026_02 — a piece of plastic in arena_2)
        ▼
Video/<date>/detections_<date>.parquet  + files_vetted_<date>.csv   pooled, analysis-ready
```

The video index **is** `file_num`, so a detection and a call from the same moment share
`start_time_real` and join directly. Detections use `location` (`arena_1` / `arena_2`), the same
vocabulary as the calls' `assigned_location`; camera names survive only in the input filenames.
`files_vetted` records which videos were actually tracked — the word *coverage* is reserved for
behavioural coverage of the gerbils. It also carries `n_stationary` (how many artifact rows that
video contributed) and `stationary_source` (`detector` = the tracking repo's flag, `fallback` = our
coarser rule for experiments it has not reached yet). These are detections, not tracks — no identity across frames,
so you can count and place animals but not follow one. Re-run with `--skip-existing` to fold in
newly tracked experiments without re-reading every CSV.

**A date folder (`2026_02`) is one continuous, weeks-long experiment on one gerbil family.**
The numbered `experiment_<id>` folders inside it are just recording restarts — not conditions.
Always analyse at the date-folder level, using `start_time_real`.

---

## Starting a new date folder

In order. Steps 2–7 are re-run as more experiments finish concatenating — the raw data usually
arrives in batches, so expect several passes.

> **Activate the venv first** — `source .venv/bin/activate`. The cluster's default `python` is the
> Lmod Jupyter module and has no pandas, so every `python` below means the repo's `.venv`.

**1. Declare it.** Add a block to [scripts/pipeline/experiments.toml](scripts/pipeline/experiments.toml).
This is the only file that needs editing; nothing else knows about date folders.

```toml
[date_folders.2026_08]
experiments = "741-815"      # ranges and single ids: "741-760, 785, 790"
skip        = [783, 784]     # optional
```

A range wider than what exists is fine — only ids you actually process are touched. The file is
validated on load, so an overlap with another folder or a stray `skip` id fails loudly.

**2. See what's ready to average.** Only experiments with a `concatenated_data_cam_mic_sync/`
folder can be processed; the rest are still upstream.

```bash
B=/mnt/home/neurostatslab/ceph/saneslab_data/big_setup
for e in $(seq 741 815); do
  [ -r "$B/experiment_$e/concatenated_data_cam_mic_sync/sync.csv" ] && echo -n "$e "
done; echo
```

Note that `big_setup` is shared — some `experiment_*` folders belong to other people, and some are
unreadable while still being written.

**3. Average the mic pairs.** Put the ready ids in `--array` in
[slurm/average-audio-array.sh](slurm/average-audio-array.sh) — the array index *is* the experiment
id, so `squeue` and the logs are labelled by experiment — then:

```bash
sbatch slurm/average-audio-array.sh
squeue -u $USER -n average-audio
```

**4. Check it landed.** Each experiment should have 3 averaged wavs per raw chunk
(`channel_{10,20,30}_file_NNN.wav`), plus `file_times.csv` and a copied experiment log:

```bash
ls .../Processed_data/Audio/2026_08/785/Averaged_wavs_w_annotations | wc -l
```

[check_audio_processing_consistency.ipynb](notebooks/ops/check_audio_processing_consistency.ipynb)
audits this against the raw channel pairs for every experiment at once.

**5. Record the cohort's light cycle — do not skip this.**
[light_cycle.py](scripts/utils/light_cycle.py) falls back to `(8, 20)` for any unknown date folder
**without warning**, which silently mislabels day/night in every circadian figure. The copied
experiment logs are the source: `grep -i "light\|dark" .../2026_08/*/experiment_*_log_*.txt`.
While you're there, add any notable events (a litter born, a manipulation) to `EVENTS_BY_DATE` in
[ethogram_io.py](scripts/utils/ethogram_io.py).

**6. Run DAS** (external to this repo) over
`Audio/<date>/<exp>/Averaged_wavs_w_annotations/`, producing
`calls_confident/<params>/*_accepted_calls.csv`. Track which experiments are done with
[find_das_completed_experiments.ipynb](notebooks/ops/find_das_completed_experiments.ipynb).

**7. Build the call tables.**

```bash
python scripts/pipeline/combine_exp_calls.py --date-folder 2026_08   # -> <exp>/calls.csv
python scripts/pipeline/pool_calls.py        --date-folder 2026_08   # -> ceph CSV + parquet
```

`combine_exp_calls` assigns each call to an arena straight from its DAS channel, which assumes no
acoustic leakage between arenas; use `run_rms_assignment.py` instead if that doesn't hold. Both
skip experiments whose DAS output is missing and say which, so running them early is harmless.
`pool_calls` rewrites the pooled CSV and parquet from scratch each time, so just run it again after
a new batch.

**8. Pool the video, if the tracking repo has run on this cohort.**

```bash
python scripts/pipeline/pool_detections.py --date-folder 2026_08 --skip-existing
```

This both puts detections on the calls' clock **and drops the detector's fixed-object detections**,
so the pooled file is analysis-ready. The per-experiment files keep every row if you need to audit
what was removed; `files_vetted.n_stationary` counts it per video.

**9. Analyse.** Every script takes the date folder by name:

```bash
python scripts/analysis/run_ethogram_categorical.py --dates 2026_08
```

Some scripts still carry a hardcoded `DATE_FOLDERS` default listing the older cohorts — pass
`--dates` explicitly, or add the new folder to that list.

---

## Where things live

| Path | What's in it |
|---|---|
| `scripts/pipeline/` | Data-*producing* steps — run these to build the parquet everything else reads |
| ├ `experiments.toml` | **The experiment-id → date-folder mapping, as data.** Starting a new date folder is an edit here, no Python change |
| ├ `audio_processing_config.py` | Reads that file and answers questions about it; also holds the mic-pair wiring and raw-filename scheme detection |
| ├ `average_audio.py` / `rms_assignment.py` | The two production pipelines (`gerbil-average-audio`, `gerbil-rms-assignment` on PATH after install) |
| ├ `paths.py` | The data roots and per-experiment path helpers — one place, no per-script copies |
| ├ `combine_exp_calls.py` / `run_rms_assignment.py` | Drivers that turn DAS output into `<exp>/calls.csv` |
| ├ `pool_calls.py` | Pools a date folder → ceph CSV + parquet cache |
| └ `pool_detections.py` | Per-frame animal detections (from the tracking repo) onto the calls' clock: `<exp>/detections.parquet` then `detections_<date>.parquet`, + `files_vetted` |
| `vocalization_analysis/` | The analysis library — imported, never run |
| ├ `bouts.py` / `acoustic_features.py` / `sync_times.py` | Bout detection · vocalpy features · audio↔wall-clock alignment |
| └ `calc_transitions.py` | Transition matrices + inter-call-gap helpers. Clean library, but `plot_transition_matrices` alone is 560 of its 919 lines |
| `scripts/analysis/` | Data-*consuming* figures and models. One script per question → see index below |
| `scripts/analysis/exploratory/` | Probes kept because committed scripts still import from them; not maintained |
| `scripts/utils/` | `ethogram_io` (**the shared loader — 14 scripts use it**), `light_cycle`, `spectrogram_viz`, `export_sync_tidy` |
| `scripts/video/` | `sync_video_spectrogram` (camera + spectrogram mp4), `play_spectrogram` (audio only) |
| `notebooks/` | All notebooks — see the index below. `ops/` = run-when-needed utilities, `archive/` = superseded, `legacy/` = pre-DAS relics |
| `slurm/` | sbatch array driver for the averaging step |
| `exports/` `figures/` `videos/` | Local outputs — **gitignored**, all regenerable |

---

## Analysis scripts, by question

### Is calling rhythmic? (circadian / ultradian / bursty)
| Script | Question |
|---|---|
| [run_autocorrelation_log.py](scripts/analysis/run_autocorrelation_log.py) | Autocorrelation with log-spaced lags — short-lag bursting *and* the 24 h peak on one axis |
| [run_call_correlogram.py](scripts/analysis/run_call_correlogram.py) | Binless correlogram straight from timestamps — bursting structure without rate binning |
| [run_call_acf_grid.py](scripts/analysis/run_call_acf_grid.py) | The ACF as a grid: one column per date, one row per call type |
| [run_call_correlogram_grid.py](scripts/analysis/run_call_correlogram_grid.py) | Same grid layout, binless correlogram version |
| [run_ici_histograms.py](scripts/analysis/run_ici_histograms.py) | Inter-call-interval distributions — all calls, then one panel per type |
| [plot_icg_histogram.py](scripts/analysis/plot_icg_histogram.py) | Inter-call *gap* histogram across date folders |

### When and where does each call type happen?
| Script | Question |
|---|---|
| [run_ethogram_categorical.py](scripts/analysis/run_ethogram_categorical.py) | Timeline strip per location, painted by the *dominant* call type per bin |
| [run_call_rate.py](scripts/analysis/run_call_rate.py) | Calls per minute per gerbil, per experiment, summarised per date |
| [run_bout_raster.py](scripts/analysis/run_bout_raster.py) | Raster of dense periods — eyeball within-area turn-taking |
| [export_bout_pages.py](scripts/analysis/export_bout_pages.py) | Bout spectrogram pages for talks |

### What follows what? (transitions)
| Script | Question |
|---|---|
| [run_transitions.py](scripts/analysis/run_transitions.py) | Per-experiment call-transition matrices from `calls.csv` + `sync.csv` |
| [run_transition_prob_by_gap.py](scripts/analysis/run_transition_prob_by_gap.py) | P(next type \| current type, gap ≈ τ) — one panel per current type, split by location |
| [run_transition_prob_by_gap_allpairs.py](scripts/analysis/run_transition_prob_by_gap_allpairs.py) | Same, but all pairs at once rather than conditioning on the current type |
| [run_transition_litter_split.py](scripts/analysis/run_transition_litter_split.py) | Before- vs after-litter transitions overlaid, per family |

### Is calling predictable from calling alone?
| Script | Question |
|---|---|
| [run_next_call_logit.py](scripts/analysis/run_next_call_logit.py) | Softmax model of the *next call's type* |
| [run_next_call_memory_ladder.py](scripts/analysis/run_next_call_memory_ladder.py) | Nested models: how much does history add beyond just the last call? |
| [run_call_rate_excitation.py](scripts/analysis/run_call_rate_excitation.py) | Self-excitation: predict *how many* calls fall in the next bin |
| [run_next_window_counts.py](scripts/analysis/run_next_window_counts.py) | Per-type counts in a window → per-type counts in the next |
| [run_switch_hazard.py](scripts/analysis/run_switch_hazard.py) | **Open pilot.** Predict *when* a bout ends, from bout age and ICI decay. If AUC ≈ chance, switching is externally driven and needs video, not audio |

> The thread through this group: the next-call model mostly rediscovers "the current bout continues".
> The memory ladder and the switch hazard exist to get past that.

---

## Notebooks

Scripts answer a question the same way every time; notebooks are where a question is still being
shaped. The dividing line: **anything the cluster runs is a script.**

### `notebooks/` — live exploration

**Bouts — does being in a bout change a call?**

| Notebook | What it does |
|---|---|
| [alarm_bouts_clean.ipynb](notebooks/alarm_bouts_clean.ipynb) | Alarm bouts via the shared `bouts.detect_bouts`; thresholds from `BOUT_THRESHOLDS["alarm"]`. The developed one (67 cells) |
| [high_freq_bouts_clean.ipynb](notebooks/high_freq_bouts_clean.ipynb) | Same for high-freq — **a 3-cell stub**, started and never filled in. HF+warble merged bouts still only exist in the archived v3 |
| [warble_singletons_vs_bouts.ipynb](notebooks/warble_singletons_vs_bouts.ipynb) | Warbles that occur alone vs. warbles inside a multi-call bout |
| [bout_transitions.ipynb](notebooks/bout_transitions.ipynb) | Bout → bout transition matrices on the 2–300 s inter-bout timescale |

**Call inventory and hand labels**

| Notebook | What it does |
|---|---|
| [explore_complex_calls.ipynb](notebooks/explore_complex_calls.ipynb) | Visual audit of the human annotations for the complex-call training set (3–4 syllable strings) — scan for label coverage gaps |
| [tmp_new_complex_folders.ipynb](notebooks/tmp_new_complex_folders.ipynb) | The two folders added July 2026: putative call *sequences* vs. calls from *two different animals* |

**Figures for talks**

| Notebook | What it does |
|---|---|
| [talk_call_examples.ipynb](notebooks/talk_call_examples.ipynb) | Best example spectrogram per call class, picked from the top-10 highest-confidence rows |
| [talk_warble_halfarc_stack.ipynb](notebooks/talk_warble_halfarc_stack.ipynb) | Every warble → half-arc → stack chain on a shared time axis |
| [busy_window_spectrogram.ipynb](notebooks/busy_window_spectrogram.ipynb) | The busiest 16 s in a date folder, as 4 rows × 4 s with DAS segments overlaid |

**Cross-platform**

| Notebook | What it does |
|---|---|
| [explore_calls_xplatform.ipynb](notebooks/explore_calls_xplatform.ipynb) | Call → call transition matrices; picks Mac-Dropbox vs. cluster-ceph paths from the host OS |

`notebooks/archive/` holds the superseded ancestors of the above (`alarm_bouts_3`,
`high_freq_bouts_v3`, `explore_alarm`, `OLD_define_alarm_bouts`, plus `alarm_bouts_exploratory`,
which is the useful one to revisit — it's the written-up record of hand-feature decoders and
side-hypotheses that **didn't** pan out).

### `notebooks/ops/` — operational utilities, run when needed

| Notebook | What it does |
|---|---|
| [find_das_completed_experiments.ipynb](notebooks/ops/find_das_completed_experiments.ipynb) | Which experiments in a date folder already have DAS output — run before a batch |
| [check_audio_processing_consistency.ipynb](notebooks/ops/check_audio_processing_consistency.ipynb) | Audits every experiment: processed folder exists, averaged files match the raw channel pairs, nothing missing |
| [combine_log_files_by_date.ipynb](notebooks/ops/combine_log_files_by_date.ipynb) | Merges the per-experiment log text files into one CSV per date folder |
| [average_audio_files.ipynb](notebooks/ops/average_audio_files.ipynb) | Interactive runner for the averaging step. ⚠ Re-implements `scripts/pipeline/average_audio.py` inline instead of importing it — the two can drift |

### `notebooks/legacy/` — pre-DAS relics, Windows-pathed

`Pre_process_A` / `pre_process_B` (file staging before DAS existed), `create_vox_csv` (builds the
DAS training csv — channel `-1`), and `Analysis__calls` (11 MB, no markdown, superseded by
`scripts/analysis/`). Read for reference; don't run without checking the paths first.

---

## Running things

```bash
source .venv/bin/activate     # required: the default cluster python has no pandas

# produce data for a new date folder (after DAS has run)
python scripts/pipeline/combine_exp_calls.py --date-folder 2026_08
python scripts/pipeline/pool_calls.py        --date-folder 2026_08

# then analyse it
python scripts/analysis/run_switch_hazard.py --dates 2026_02 --format png
```

Near-universal conventions in `scripts/analysis/`: `--dates` (one or more date folders),
`--out-dir`, `--format {pdf,png}`. Figures land in `exports/`.

Use the repo `.venv` for everything. The cluster's default `python` (the Lmod Jupyter module) and
the registered `gerbil-vox-jupyter` kernel both lack pandas, and fail with `ModuleNotFoundError`.
Every script picks its ceph vs. SMB base path automatically from `platform.system()`.

---

## Gotchas worth knowing before you trust a figure

- **Per-location rates conflate calling with occupancy.** A quiet burrow may just be an empty
  burrow. Treat location splits as exploratory until video tracking lets you divide by occupancy.
  The whole-colony rhythm and the newborn-litter surge don't depend on this.
- **Audio and video clock-*drift* apart** (~0.07%, audio runs fast) in
  `concatenated_data_cam_mic_sync`. It is not a constant offset. `sync_video_spectrogram`
  computes the ratio from ffprobe durations; anything else combining audio + video must too.
- **Not all acoustic features are equal.** `peak_freq` and `max_pitch` are trustworthy on gerbil
  USVs; entropy / FM / goodness are noisy because YIN drops out on fast frequency sweeps.
- **Log-binned histograms need `density=True`** — and for heavy-tailed data, prefer
  log10-transforming the values with linear bins over log-spaced bins on a log axis, so the tail
  stays visible.
- **`calls.csv` is built without cross-channel dedupe** for 2025_07 / 2025_10 / 2026_02, which
  assumes no acoustic leakage between arenas. It holds for those datasets; check before assuming
  it holds for new ones.

---

## Known rough edges

- **Two scripts write `<exp>/calls.csv`** — `combine_exp_calls.py` (channel → arena) and
  `run_rms_assignment.py` (RMS + dedupe). Whichever ran last wins, and the file records no
  provenance. Current datasets use the first.
- **`get_experiment_audio_dir` and `BASE_PROCESSED` are still copied across the analysis scripts.**
  `scripts/pipeline/paths.py` is now the single home; the pipeline uses it, the analysis scripts
  don't yet.
- **`average_audio_files.ipynb` re-implements `scripts/pipeline/average_audio.py` inline** rather than
  importing it — two copies of the averaging logic, free to drift.
- Analysis scripts still start with a hand-rolled `sys.path` block. `scripts/` and
  `scripts/pipeline/` are now installed packages, so pipeline code imports by name; give
  `scripts/analysis/` and `scripts/utils/` an `__init__.py` and those blocks go away too.
- `calc_transitions.plot_inter_call_gap_distribution` has no callers left.
- Notebook outputs are committed, which is why `.git` is ~500 MB.
