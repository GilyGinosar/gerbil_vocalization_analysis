# Do gerbils call when they cross the burrow tunnel? — handoff

Written 2026-08-23, substantially revised 2026-09-01. Everything below is measured on
**2026_02** unless stated. **Read the next section first** — it supersedes parts of what
follows, and says which parts.

## The question, and the short answer

Does calling accompany an animal moving between the nest and the arena?

**Yes, and the event is ARRIVAL AT THE NEST, not tunnel entry.** USV rate (high-freq +
warble) peaks at **~2.4 calls/s at the moment a `to_nest` animal leaves the tunnel**,
against a ~0.7/s baseline, decaying over ~10 s. A `to_arena` animal arriving at its own
destination shows **nothing** — flat at chance on the same mics, same detector, same
pipeline. That contrast is the main evidence the effect is real rather than an artifact
of movement noise or of listening at one end.

Every earlier attempt anchored on tunnel *entry* and looked weak or null. Anchoring on
entry smears the arrival across the traverse-duration spread (median 1.0 s, tail to tens
of seconds); anchoring on exit sharpens it.

## Current state, 2026-09-01 — read this first

### The result

**The arrival burst needs residents PRESENT, but does not care whether they are awake.**

```
              n    any call    burst          95% CI
  empty     293      39%       0.336     [+0.25, +0.43]
  sleeping 1548      72%       1.460     [+1.37, +1.55]
  active   1856      80%       1.393     [+1.30, +1.48]

  empty vs sleeping   within-experiment   +1.106   p < 0.0001   (33 experiments)
  sleeping vs active  within-experiment   -0.075   p = 0.66
```

`burst` = underground call rate in the arrival window (`t_out - 0.5` to `t_out + 2.0`)
minus a baseline of equal length (`t_entry - 8` to `t_entry - 5.5`), while the animal is
still in the arena. The subtraction is the whole point: an active nest calls more *all
the time*, and the first version of this analysis reported that background as an arrival
effect.

`sleeping` and `active` differ **20x in motion** and give the same burst; `empty` and
`sleeping` are **indistinguishable by motion** (0.0002 vs 0.0007, both at the sensor
noise floor) and differ **fourfold**. So the variable is presence, not activity. No
motion measure can see this distinction — only looking at the frames can, which is why
the categories are hand-scored.

The within-experiment permutation only ever compares empty against occupied *inside the
same experiment*, so DAS threshold drift, colony composition, IR gain and date all
cancel. The effect is marginally larger under it than without.

**What to concede before being asked.** Calls still occur into an apparently empty nest —
39% of them, burst +0.34 with a CI excluding zero — so the claim is *four times smaller*,
not absent. "Empty" means **no animal visible**; one under bedding is invisible, which
dilutes the contrast rather than manufacturing it. And `active` (n=1,856) is *assumed*
occupied because something moved — nobody has verified it, and that is the mirror of the
assumption that proved wrong for "still".

### Why the labels can be trusted

1,891 traverses scored by eye. **1,363 of them blind** — an undergrad scored the whole
remaining set in a grid that deliberately hides the call count, so those labels cannot
have been nudged by the outcome. Blind and non-blind give the same effect (+1.11 vs
+1.14).

Then **all 312 traverses called empty were re-checked** in a 460 px grid: **2 needed
correcting, 0.6%**. Early spot checks of single large frames had suggested ~17%, so the
honest statement is that the empty set is verified end to end, not sampled.

Every correction across the whole project — 9 of them — has run the **same direction**,
occupied mislabelled as empty, and each has moved the effect **up**: +1.01 → +1.09 →
+1.11. Hand-scoring is biasing the result *against* itself.

### The never-usable rules are now enforced in code

`scripts/utils/data_rules.py` holds them and applies them **by default**:

- **the truncated last chunk of each experiment** (~1.2% of calls, ~2.3% of traverses).
  This was documented in README and in `ethogram_io`'s docstring and was still missed —
  the nest-motion analysis ran for two days on those chunks. A rule in prose does not
  propagate; a default argument does.
- **a capped `t_out`** — `burrow_scan` returns `t_exit + MAX_LINGER_S` when the tunnel
  never reads empty, an invented number up to 5 s late. Anything anchored on arrival must
  drop it.

`load_all_calls` and `data_rules.load_traverses` both apply them, with `keep_*` flags to
opt out deliberately. `scripts/utils/check_direct_reads.py` finds code that reaches past
the loaders to `pd.read_parquet` — in notebook cells as well as `.py` files. **25 such
reads remain**, mostly older notebooks whose published numbers came from the old
behaviour; they are flagged, not silently changed.

### Where things are

| what | where |
|---|---|
| the labels, corrections, both scorers | `data/nest_scoring/` — **tracked**, was in gitignored `exports/` |
| the 3,786-traverse motion run | `data/nest_scoring/nest_motion_full.csv` |
| the row set the undergrad saw | `data/nest_scoring/undergrad_rows.csv` — **needed**: her export lists only what she TICKED, so empties are knowable only by difference |
| finished figures | `exports/<run date>/` and `Combined/<date folder>/`, both, nothing overwritten |

Three one-command drivers, each encoding a flag combination where the defaults are wrong
in ways that change what the figure says:

    python scripts/analysis/make_burrow_overview.py --out-dir <dir>
    python scripts/analysis/make_traverse_cards.py --category empty --out-dir <dir>
    python scripts/analysis/traverse_time_of_day.py --scan <scan> --out-dir <dir>

### Bugs found in figures that had already been shown

Worth knowing because each one changed what a figure said, and each was invisible:

- **`burrow_cards` drew every call backwards.** `to_nest` cards got left-to-right travel
  by reversing the *time axis*, which mirrored the spectrogram with it — a rising USV
  rendered as falling. Now the tiles' pixels are mirrored and time runs forward.
- **The raster y-limit was a fixed ±20 rows.** 1% of a 3,786-row panel, but **44% of a
  44-row panel**, where it read as traverses with no calls. Now proportional.
- **Capped-`t_out` traverses were in every figure**, anchoring the arrival window up to
  5 s off the event.
- **Card end panels shifted the time axis.** Constant width whenever a panel existed, but
  a 320 px black fallback when a frame could not be read — so most cards aligned and the
  odd one silently did not. Now fixed-width; measured 1 px spread over 20 cards.
- **`arena_occupancy_by_hour` was committed in a state where it could not run** — calling
  `load_traverses` and `publish` without importing either.

### Other results from this stretch

**A new litter arrives 2026-03-02 and the traverse rate halves** — 29/h before, 13/h
after. Dated independently by the logs ("mom making nest around newborns") and by newborn
calls going 1.7% → 25.7% of all calls. It is a **step, not a slope**: within either period
neither elapsed day nor call intensity predicts the rate. Any analysis pooling 2026_02
averages two behaviourally different regimes.

**The morning traverse peak is mostly occupancy.** Arena occupancy swings 9x across the
day while traverses swing 4x. Normalise by the pool that could make each crossing and it
inverts: emergence peaks in the morning, while the few animals out after dark return
fastest. Tracking only reaches experiment 521 (Feb 28), so this is **entirely pre-litter**
and cannot speak to what the litter changed.

**For scheduling a logger:** peak 39.6/h at 07:00, +3 h after lights on, and the peak hour
wins at every window length up to 7 h. But the day-to-day spread is 30–60% (the error
bands bootstrap over *days*, not events), so a short fixed window is a gamble on any one
day.

## The pipeline, in order

| step | script | what it does |
|---|---|---|
| 1 | `scripts/video/burrow_scan.py` | Scans whole `burrow_side` videos. Background-subtracts, counts animal blobs, writes a per-frame track (`n_animals`, `x`, `area`, `moved`) and detects traverses as landmark crossings. Caches a frame strip per traverse. |
| 2 | `scripts/video/pool_scan.py` | Pools the per-video CSVs into `traverses_<date>.parquet`, adds `exp` and wall-clock time. |
| 3 | `scripts/analysis/raster_and_rate.py` | Every figure, from one collection pass: rasters + rate curves aligned to entry, to exit and by position; the within-file shuffle null (`--shuffle`, on by default); and the nest-end vs arena_1 mic control (`--by-mic`) that makes the result nest-specific rather than a missing microphone. Caches the collection to `collected.npz`; `--recollect` rebuilds. |
| — | `scripts/video/burrow_cards.py` | Curation cards from cached tiles, stacked into JPG contact sheets. `--position-band lo,hi` selects by where in the tunnel the calls happened; `--channels 1,0` draws the tunnel mic above the nest mic; `--localiser-marks` adds a ribbon of each call's tunnel/nest verdict; `--nest-frame` welds a rotated nest_top frame from the moment of entry; `--prior-nest yes/no` selects on whether the nest had been calling. |

| — | `scripts/analysis/localiser_threshold.py` | **Read this before trusting any tunnel/nest label.** Draws the two dB populations and their overlap, what the label is made of (mostly false alarms), the precision/sensitivity trade, and the position gradient that shows the cut is really a position cut. |
| — | `scripts/analysis/burrow_overview.py` | Every condition on one page: call set x alignment x condition, rasters and rates. `--shuffle 0` by default — it is for looking. |
| — | `scripts/analysis/nest_motion.py` | Did anything move in the nest before arrival (video, independent of the audio). Windowed pilot; see the nest-occupancy section. |
| — | `scripts/analysis/nest_occupancy_examples.py` | nest_top frames grouped by the census count, for eyeballing. The census is not trustworthy yet — see below. |

These four figure scripts are the whole analysis surface. `entry_psth.py`, `call_rate_by_position.py`,
`arrival_by_mic.py`, `sample_cards.py`, `picker_to_sheets.py` and `scan_cards.py` were
folded into them on 2026-08-23 and moved to `scripts/archive/`; their figures are all
reproducible from the two survivors.

Cluster: `slurm/burrow-scan-disbatch.sh <taskfile>`, task files built by
`make_scan_tasks.py`. The full 2026_02 scan is 3,775 videos, ~2.5 h on one exclusive
`gen` node. Output lives on ceph at `/mnt/ceph/users/gginosar/burrow_scan_2026_02`
(3.7 GB: tracks, tiles, CSVs).

## What the numbers are

- 3,775 videos scanned, 77 rejected as corrupt, **8,627 traverses** (8,319 single-animal),
  4,168 `to_nest` / 4,151 `to_arena` — near-perfect balance, unlike the old motion
  detector's 12/17 which was a candidate-proposal artifact.
- **Clean transits** (tunnel empty 3 s before entering the ROI and after leaving):
  2,749 `to_nest`, 2,791 `to_arena`. The filter takes both directions equally (66% / 68%).
- **Arrival burst**: `to_nest` 2.4 calls/s vs ~0.7 chance. `to_arena` at its own arrival:
  flat. On arena_1 mics, `to_arena` arrival is 1.18x chance — real but ~20x smaller than
  the nest effect, and a slow step rather than a spike.
- **Duration asymmetry** (independent of the call analysis): `to_arena` takes about twice
  as long as `to_nest` — 2.89 s vs 1.41 s median for strictly-alone traverses. Survived
  every filter and grew when social encounters were removed.

## What is NOT established — read this before building on anything

**"Tunnel-origin" does not mean "from the tunnel". It means "arena-half of the
tunnel".** This is the most important correction in the file — measured 2026-08-26,
and it invalidates any reading of `localise_calls.py` as a compartment split.

For calls made while exactly one animal was in the tunnel, join each call to that
animal's tracked position and the level difference turns out to track WHERE IT WAS
STANDING (20 experiments, 16,274 calls):

| animal at x | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.4-0.5 | 0.5-0.6 | 0.9-1.0 |
|---|---|---|---|---|---|---|
| median dB | -8.13 | -7.12 | -3.50 | -0.10 | +1.65 | +4.86 |

x=0 is the nest end. The nest-origin reference has median -5.6 dB, so **an animal
genuinely inside the tunnel, in its nest-end half, scores like the nest** and is
labelled nest-origin. Nest half (x<0.3) -6.81 dB vs arena half (x>0.6) -0.07 dB: a
**6.7 dB position gradient, twice the 3.1 dB occupied-vs-empty shift the localiser
is built on**. Three consequences:

- the tunnel-origin series is sparse and its absolute height is meaningless;
- the collapse of tunnel-origin at arrival is at least partly geometric — a
  `to_nest` animal spends its last second at the nest end, outside the label's
  sensitive zone, so it stops being counted before it stops calling;
- the arrival burst is partly the ARRIVING animal, not only residents: it scores
  nest-origin from about x<0.3, i.e. before it leaves the tunnel at all.

Use tracked position, not the dB threshold, wherever an animal is in the tunnel.
`scripts/analysis/localiser_threshold.py` draws the gradient and the base-rate
problem below.

**The label is also mostly false alarms.** The tunnel-empty population is 11x
larger than the occupied one, so at the default q=0.95 cut its 5% tail (20,437
calls) outnumbers the genuine hits (14,965): **only 42% of "tunnel-origin" calls
had anyone in the tunnel.** `--localiser-quantile 0.99` takes precision to 69% at
the cost of sensitivity (41% -> 25%). Note the "5.0% of empty-tunnel calls pass"
line the script prints is the definition of the 95th percentile, not a result.

**The threshold drifts within an experiment.** It is calibrated per experiment,
which absorbs any fixed gain mismatch between ch01 and ch00 — the classification
never uses the absolute sign, so a constant offset cancels. But the tunnel-empty
median wanders 1.8 dB sd / 7.0 dB spread across files WITHIN one experiment,
against a 3.1 dB signal. Per-FILE calibration is the obvious fix and has not been
tried. The absolute gain difference cannot be recovered post hoc without a known
source; the playback logs remain the outstanding TODO.

**The origin-split result was mostly relabelling.** An earlier framing used the two
underground mics to localise each call and reported 8.5x at entry. The dwell control
(`dwell_control.py`, exp 492) gives **0.277 calls/s for transits and 0.277 for
non-crossing dwells — identical**. **The full dwell control across all 60
experiments has NOT been run** — `slurm/dwell_2026_02.tasks` is built and ready.

**Two tests that cannot work, and why.** "Calls inside the tunnel period vs outside"
returns ~1.0 by construction, because calling starts before entry and continues after
exit. And requiring an empty tunnel just before the landmark crossing rejects 100% of
traverses, because the animal is already inside the crop before it reaches the landmark —
the clean-transit test has to bracket the ROI occupancy run, not the landmark.

**What survives the localiser correction, and what does not.** The raw count needs
no label at all: underground calls go from **0.51 calls/s during the traverse to
1.70 in the 2 s after the animal is out** — and `--clear 3.0` guarantees the tunnel
is empty for 3 s after exit, so those calls cannot come from the tunnel. That
tripling stands whatever the localiser does. The compartment comparison
(arena_1 vs underground) also stands: those are separate mic arrays, and
`qc_orphan_calls.py` records only **3% of arena_1 calls in minutes with no arena_1
detection** (vs 43% for arena_2), so arena_1 assignment is ~97% consistent with
video. What does NOT stand is any absolute reading of the tunnel-origin height, or
treating tunnel/nest as two clean sources.

**Prior nest calling predicts calling in the tunnel, and it is not just bouts.**
Split `to_nest` traverses by whether any nest-origin call occurred in the 5 s
before entry. Tunnel-origin rate while in the tunnel: **0.627 calls/s with a prior
nest call vs 0.310 without (2.0x, p<1e-4)** at q=0.95; 0.299 vs 0.100 (3.0x) at
q=0.99. A placebo "prior" window 30-35 s earlier gives only 1.33x, and the adjacent
effect survives conditioning on it (+0.294 among traverses with no distant prior,
+0.327 among those with one, both p<1e-4). On USV only at q=0.99 the effect is
**3.97x with a placebo control that is not significant (p=0.10-0.17)** — the
cleanest version. Caveat: the position confound above means "tunnel-origin rate"
here is partly "how far along the tunnel it was", so read this as an association
between prior nest calling and calling during the crossing, not as a source claim.

**The arrival burst does not need the nest to be active.** With zero nest calling
in the preceding 5 s, arrival still gives **1.503 calls/s vs 2.230 when the nest
was already calling** — two-thirds of the effect survives with the bout confound
removed by construction. Note the "silent" group means DAS logged nothing, not that
the channel was quiet.

**Light/dark makes no difference to the arrival effect.** 2026_02 runs lights
04:00-16:00 (`scripts/utils/light_cycle.py`, NOT the 08:00-20:00 default).
Arrival is 2.84x in light vs 2.62x in dark. Arena calling is higher in daylight
overall (0.354 vs 0.227 calls/s at arrival, 0.206 vs 0.143 baseline), but the
proportional arrival boost does not differ (1.72x vs 1.59x, p=0.53) — the additive
difference that looks significant (p=0.026) is that multiplicative effect riding on
a bigger baseline.

**Call types: use all of them for compartment comparisons.** underground is 69% USV
/ 29% stacks; arena_1 is 95% USV / 4% stacks. Counting USV-only on one side and
everything on the other is not a like-for-like comparison. In practice it barely
matters near a traverse (+6% underground, +4.5% arena_1 on the same traverses), but
for the prior-nest question the non-USV classes ARE the bout control's significance
— `--types usv` gives a null placebo, `--types all` gives p<0.01.

**Shuffle p-values are optimistic.** 40% of consecutive traverses are less than 20 s
apart, so their ±10 s windows overlap and the same call is counted for several anchors.
The shuffle treats traverses as independent. Shapes are fine; significance is overstated.

## Known data problems

**DAS type labels are unreliable between high-freq and warble.** Their PSTHs correlate at
r = 0.945, and on individual cards a train of visually identical sweeps gets labelled
alternately orange/magenta. **Always pool them.** Do NOT pool in `stacks` — it
anti-correlates (r = −0.745), is a third of all calls, and cancels the signal. `newborn`
(7.6%) is presumably pups and is a confound for a nest-arrival question.

**Compartment assignment is good in aggregate, imperfect near the tunnel.** Colouring
raster ticks by assigned compartment shows a sharp handover at the tunnel, so it tracks
the animal — but individual cards show calls on the wrong row near the mouth.

**Audio/video drift is per-file, not per-cohort.** 2026_02 is clean (+1 ms). 2025_03 /
2025_07 / 2025_10 vary file to file by up to ~900 ms and change sign, and it is dropped
*video frames*, not a clock — so a uniform stretch cannot fix it. `frame_id/30` is not
trustworthy there at sub-second scale. The README gotchas section has the measurements.

**The centroid cannot reach the crop edges.** A body centre saturates at ~0.12 (nest end)
and ~0.77 (arena end) as the animal leaves frame, which produced a spurious narrow band in
the position raster. `raster_and_rate.py` now drops the last 0.3 s (`SATURATE_S`) and uses
the whole ROI visit rather than the landmark window, giving 0.08–0.94 coverage for both
directions.

**Short-audio chunks are a real hazard but immaterial here.** Parallel work found 41
chunks in 2026_02 where video outruns audio by >1 min. Only **10 of 8,319 traverses
(0.1%)** sit in chunks with a short wav — the corrupt-video guard had already excluded the
badly broken ones. Re-check this if the guard is ever loosened.

## Who is calling: the nest-occupancy thread (2026-08-25/26)

The localiser cannot answer "residents or the arriving animal" — both are
underground, and after exit the traveller IS in the nest. Three routes were tried.

**Counting animals in the nest from video: dead end.** There is no detector on any
nest camera (`detections.parquet` covers `arena_1` and `arena_2` only, from the two
`video_center` cameras) and the gerbils burrow under the bedding, so absence of a
detection is exactly what a buried animal looks like. Neither frame differencing
nor YOLO fixes that — it is occlusion, not an algorithm.

**Census by subtraction: built, and it does not survive inspection.** Nest =
colony − arena detections − the traveller. Colony size is now recorded:
`n_animals = 6` for 2025_07, 2025_10 and 2026_02 in `experiments.toml`, read via
`get_colony_size()` (returns None when unknown — never treat that as zero). The
distribution spreads usefully across 0–5 residents, BUT **7% of traverses give a
negative count**, which is impossible and means the arena detector over-counts.
The cause is visible in `exports/burrow/arena1_crowded/`: a huddle gets split into
4–5 overlapping boxes. Huddles are only 0.8% of all frames but far commoner during
traverses, because traverses happen when the colony is active. Gily eyeballed
`nest_occupancy_examples.py` output and judged the census "not doing a good job" —
the impossible rows show busy nests, i.e. it under-reports the nest. An IoU merge
of overlapping boxes is the obvious fix and has NOT been tried.

**Nest MOTION rather than nest counting: the promising one, and unfinished.**
`nest_motion.py` asks only whether anything moved in the nest, which is far easier
than counting and — the reason it matters — **independent of the audio**. The
prior-nest-CALL split is close to circular when the question is whether the nest
calls; a video motion split is not. For a `to_nest` traverse the animal is in the
arena before entry, so pre-entry motion is residents only. A single-file prototype
gave a wide dynamic range (median 0.021 of pixels changing per frame, p95 0.077,
one traverse with literally 0.0000 before entry), and motion rose after arrival in
7 of 8 traverses — the traveller entering view, which is why only the PRE-entry
window is used for the split.

**RESULT (400 traverses, 2026-08-26).** Pre-entry nest motion is heavily skewed
towards stillness — median 0.0032 of pixels changing per frame, p25 0.0001, so the
lower quartile is a nest where essentially nothing moves. Split at the median:

| pre-entry nest | n | calls/s BEFORE entry | calls/s AT ARRIVAL | arrival / before |
|---|---|---|---|---|
| QUIET | 200 | 0.382 | 1.260 (59% had any call) | 3.30 |
| ACTIVE | 200 | 0.697 | 1.740 (78% had any call) | 2.50 |

active − quiet: **+0.480 calls/s at arrival (p=0.003)**, but **+0.315 calls/s
before entry too (p<1e-4)** — so a moving nest is a noisier nest generally, and the
arrival difference is riding on that. Normalising by each group's own pre-entry
baseline REVERSES the ranking: the arrival burst is **3.30x baseline when the nest
was still, 2.50x when it was already active**. Read that carefully before building
on it — the multiplicative-vs-additive trap that already caught the light/dark
analysis in this file.

Two things it does establish. **The burst happens from a still nest** — 1.260
calls/s and 59% of traverses with a call when nothing was moving beforehand — which
matches the audio-based silent-nest result (1.503 vs 2.230) and argues the arriving
animal contributes on its own. And this is the **first evidence on the residents
question that is not circular**: nest motion is measured from video, the burst from
audio, so unlike the prior-nest-CALL split it does not use calling to predict
calling. Whether an active nest adds anything arrival-specific is NOT settled —
the proportional analysis says no, the additive says yes, and 400 traverses is a
pilot. Re-run or extend with:

    python scripts/analysis/nest_motion.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir exports/burrow/nest_motion --limit 400

It takes ~35 min for 400 traverses (per-traverse seeking dominates). If pre-entry
motion predicts the arrival burst, the full scan is worth it: 3,775 `nest_top`
videos, exactly matching the 3,775 `burrow_side` videos, ~2.5 h on one exclusive
node under disBatch — same shape as `burrow_scan.py`. Mask the tunnel mouth before
using any POST-arrival motion.

## Should the top camera replace the side camera for tracking? Probably not yet

`video_burrow_top` exists for every file. On one traverse it looked clearly better;
across three experiments it is a draw — better where bedding is shallow, no better
where it is deep, and the framing varies between experiments so it needs its own
per-experiment geometry. A mesh grid sits on the tube and would confuse frame
differencing. Examples in `exports/burrow/camera_compare/`.

Against that, the existing side-view tracking was good enough to reveal the 6.7 dB
position gradient, and the centroid saturation it is blamed for is milder than the
docs imply: one spike at x 0.08–0.12 (~9% of frames), otherwise a smooth spread,
reaching 0.007 to 0.968. The real blocker is "who is calling", which better
position does not resolve. Revisit only if you need the last 10% of the tube.

## Open questions, in the order I would take them

1. **Verify the `active` category.** 1,856 traverses assumed occupied because something
   moved, never looked at. It is the same assumption that proved wrong for "still", and
   a few hundred scored in the grid would settle it. `nest_grid_picker.py` builds the
   page; give it its own `--storage-key`.

2. **An inter-rater number.** The undergrad's set and Gily's do not overlap by design, so
   there is no agreement figure. A few dozen deliberately shared rows would give one, and
   it is the natural answer to "how reliable is eye-scoring".

3. **Extend tracking past experiment 521.** The raw arena video exists for all 67
   experiments; tracking stopped at 521 (Feb 28), one day before the litter. Running it
   over 522–567 is the only way to test whether the litter kept animals indoors — the
   obvious mechanism for the traverse rate halving, currently untestable.

4. **The DAS cross-channel bug is still unfixed.** `exports/MESSAGE_TO_DAS_PIPELINE.md`
   documents it, is hand-written, and is the only copy in a gitignored folder — move it
   into `docs/` and commit it. It preferentially deletes the loudest copy of *underground*
   calls (41 of 53 audited) in favour of leaked arena copies. That is the compartment this
   whole analysis counts.

5. **Cross-talk vs the no-leakage assumption.** That same message documents real
   cross-talk in exp 492 (274 sub-15 ms cross-location pairs, ~22% of arena_2 coinciding
   with arena_1), while the stored assumption says 2026_02 has none and `calls.csv` is
   built without dedupe on that basis. Both cannot be true.

6. **`single_animal` has false negatives.** Two of fifteen inspected cards showed a second
   animal the tracker had not flagged (`multi_animal_frac = 0.000`). That column gates
   every analysis here and has never been audited.

7. **Who calls — the remaining routes.** The nest-occupancy split has gone as far as it
   can. TDOA between ch01 (tunnel mouth) and ch00 (deep nest) measures arrival geometry
   directly rather than inferring it from level, which is what made the dB threshold a
   position gradient. Note the ~0.07% audio/video clock drift is irrelevant between two
   audio channels but would swamp a millisecond-scale delay if video is used as an anchor.

## Practical notes

- Tunnel mic is **raw channel 01**, 0-based; underground pair is {0,1} for exp >= 272.
  Verified: +7.2 dB in-transit on ch01 vs +2.9 on ch00 and ~0 on arena mics.
- Decoding is ~94% of the cost of everything. Once a frame is decoded, every per-frame
  measure is nearly free — compute them all in one pass. The frame difference costs +0.5%.
- Cached tiles are why curation sheets rebuild in seconds. Do not lose them.
- `raster_and_rate.py` caches its collection to `collected.npz`; `--recollect` rebuilds.
- Figures live under `exports/` (gitignored, all regenerable).
