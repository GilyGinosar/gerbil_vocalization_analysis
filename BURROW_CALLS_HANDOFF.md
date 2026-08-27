# Do gerbils call when they cross the burrow tunnel? — handoff

Written 2026-08-23. Everything below is measured on **2026_02** unless stated.

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

**Status: the 400-traverse pilot was still running when this session ended.** It
writes `exports/burrow/nest_motion/nest_motion.csv` and prints a median split with
a permutation test. Re-run or resume with:

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

1. **Who is calling** — the arriving animal or the nest residents greeting it. First-call
   positions cluster at the same absolute position (~0.35) in both directions, which hints
   at a place effect and would be consistent with residents. The localiser was built for
   this and now has a well-posed question.
2. **Run the full dwell control.** One command, already prepared.
3. **The position curve is U-shaped for `to_nest`** — highest at the arena end (1.9),
   dipping mid-tunnel (0.73), rising again near the nest. Earlier framings that described
   a monotonic climb toward the nest were reading a truncated axis. Unexplained.
4. **Inter-call intervals.** The nest-mouth examples are steady trains of near-identical
   sweeps, not bursts. `vocalization_analysis/bouts.py` already exists.
5. **Playbacks are still counted as calls** (see the repo's own gotchas) and have not been
   excluded anywhere in this work.

## Practical notes

- Tunnel mic is **raw channel 01**, 0-based; underground pair is {0,1} for exp >= 272.
  Verified: +7.2 dB in-transit on ch01 vs +2.9 on ch00 and ~0 on arena mics.
- Decoding is ~94% of the cost of everything. Once a frame is decoded, every per-frame
  measure is nearly free — compute them all in one pass. The frame difference costs +0.5%.
- Cached tiles are why curation sheets rebuild in seconds. Do not lose them.
- `raster_and_rate.py` caches its collection to `collected.npz`; `--recollect` rebuilds.
- Figures live under `exports/` (gitignored, all regenerable).
