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
| — | `scripts/video/burrow_cards.py` | Curation cards from cached tiles (seconds, no video decode), stacked straight into JPG contact sheets for viewing over SSH. `--position-band lo,hi` selects traverses by where in the tunnel their calls happened. |

These two are the whole analysis surface. `entry_psth.py`, `call_rate_by_position.py`,
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

**The origin-split result was mostly relabelling.** An earlier framing used the two
underground mics to localise each call (`localise_calls.py`) and reported 8.5x at entry.
The dwell control (`dwell_control.py`, exp 492) gives **0.277 calls/s for transits and
0.277 for non-crossing dwells — identical**. While an animal is in the tunnel its calls
get labelled tunnel-origin regardless of rate, which produces a peak by bookkeeping. The
total-rate result above avoids this entirely; the origin split does not. **The full dwell
control across all 60 experiments has NOT been run** — `slurm/dwell_2026_02.tasks` is
built and ready.

**Two tests that cannot work, and why.** "Calls inside the tunnel period vs outside"
returns ~1.0 by construction, because calling starts before entry and continues after
exit. And requiring an empty tunnel just before the landmark crossing rejects 100% of
traverses, because the animal is already inside the crop before it reaches the landmark —
the clean-transit test has to bracket the ROI occupancy run, not the landmark.

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
