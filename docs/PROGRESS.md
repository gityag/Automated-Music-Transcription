# Progress Log

Dated entries, most recent first. This is the raw material for the
capstone report's methodology section -- write here as you go, not
after the fact.

---

## Week 1 -- Pipeline foundations

**Goal:** replace the buggy hand-rolled MIDI/CQT pipeline with a typed,
tested foundation. Nothing downstream (training, evaluation, results)
is meaningful until this is solid.

**Built:**
- `amt/config.py` -- one typed `Config` dataclass (audio, CQT, labels,
  split, model, train, eval settings) replacing constants that were
  previously hardcoded and duplicated across scripts (e.g. `lowest`/
  `highest` re-assigned inside `encode_midi_segments.py` regardless of
  what was passed in). Validates that the CQT pitch range and the label
  pitch range agree -- see [ADR 0003](decisions/0003-frame-rate-and-pitch-range.md).
- `amt/data/midi.py` -- MIDI parsing via `pretty_midi` plus an explicit,
  tested sustain-pedal extension rule. Replaces ~300 lines of hand-rolled
  tempo/tick math. See [ADR 0001](decisions/0001-pretty-midi-for-labels.md).
- `amt/features/cqt.py` -- log-magnitude CQT, replacing the real-part-only
  feature. See [ADR 0002](decisions/0002-log-magnitude-cqt.md).
- `amt/utils/seed.py` -- one seeding call for numpy/random/torch.
- 18 unit tests (`tests/`), several written specifically as regression
  tests for bugs found during the audit: held-note false gaps,
  re-struck-note merging, phase sensitivity of the old CQT feature,
  and config-level pitch-range mismatches.

**Verified:** `pytest tests/` -- 18 passed.

**Bugs found and fixed this week** (see linked ADRs for detail):
1. Single-tempo assumption in MIDI parsing.
2. Re-struck notes at the same pitch silently merged.
3. Synthetic note-offs at segment boundaries creating false ~83 ms gaps
   in held notes.
4. CQT feature used only the real part, which is phase-dependent noise
   rather than musical signal.
5. `ref=np.max` normalization made a note's feature value depend on
   what else was in its clip.
6. CQT input pitch range (MIDI 24-107, librosa default) silently
   disagreed with the label pitch range (MIDI 21-107).
7. Time resolution (~83 ms/bin) exceeded the ±50 ms tolerance the
   system would later be evaluated against.

**Not yet done:** dataset assembly (song-level train/val/test splitting
against MAESTRO), the PyTorch data loader, the model, and the
`mir_eval`-based evaluation script. These are next.

**Next session:** `amt/data/dataset.py` (song-level split against the
official MAESTRO metadata CSV, feature/label extraction and caching to
disk) and `amt/evaluation/` (frame- and note-level metrics via
`mir_eval`).

## SMD validation, two label bugs found and fixed

**Goal:** smoke-test the Week 1 pipeline (`amt/data/midi.py`,
`amt/features/cqt.py`) against real audio+MIDI, starting with one file
by hand, then all of SMD's 50 pairs via a new validation script.

**Built:** `scripts/validate_smd.py` -- pairs every `data/smd/midi/*.mid`
with its matching `data/smd/midi_wav_22050_mono/*.wav` by filename stem,
runs the full pipeline (load audio -> CQT -> sustain-pedal-extended
notes -> piano-roll labels) on each pair inside a try/except, and
reports per-file shapes or failures.

**Bugs found by running it:**

1. **Frame-count mismatch.** `notes_to_piano_roll` computed its own
   frame count as `ceil(duration * fs)`. librosa's CQT computes its
   frame count as `1 + samples // hop_length`. These agree only when
   the division isn't exact -- on SMD, 8/50 files had CQT and label
   shapes off by one frame. Root cause: two independent formulas for
   the same quantity, rather than one shared source of truth.

   Fix: changed `notes_to_piano_roll`'s signature from `duration:
   float` to `n_frames: int`, so the caller passes `cqt.shape[1]`
   directly instead of the function re-deriving its own (inconsistent)
   frame count. `duration` is still needed internally for clipping
   (below) and is recovered as `n_frames / fs`.

2. **Crash on unresolved sustain pedal.** `Rachmaninoff_Op036-02`
   failed with `cannot convert float infinity to integer`. Cause:
   `extend_notes_with_sustain_pedal`'s pedal-down interval can be
   `(down_time, inf)` when the pedal is never released before the file
   ends (see `_pedal_down_intervals`), so a note ending inside that
   interval got `end = inf`, which crashed `int(np.ceil(note.end *
   fs))` in `notes_to_piano_roll`.

   Fix: clip the note's end to the recording length (`min(note.end,
   n_frames / fs)`) before it's used in any arithmetic.

3. **Defensive fix, not from a crash:** `start_frame` had a lower
   clip (`max(0, ...)`) but no upper clip, so a note starting at or
   past the recording's end could index one past the array's bounds.
   Added `min(..., n_frames - 1)`. No SMD file triggered this, but a
   MIDI file with a stray note past the audio's length would have.

**Tests added:** a regression test per bug in `tests/test_midi.py`
(19 tests total, all passing), including one that builds a note
starting deliberately past `n_frames` and checks it clips into the
last valid frame rather than crashing.

**Verified:** re-ran `scripts/validate_smd.py` after the fix --
50/50 files now produce matching CQT/label shapes, including the file
that previously crashed.

**Lesson for the report:** all three bugs were invisible from reading
the code in isolation -- each only showed up by running the pipeline
against real files and checking shapes/errors, not from inspection.
This is the argument for why the validation script exists as a
standing artifact, not a one-off scratch check.

---

## Shared process_song() function + feature caching

**Goal:** eliminate duplicated pipeline logic between scripts (the
original sin behind Friday's two bugs shipping unnoticed for a while),
and stop recomputing each song's CQT/labels from scratch on every use.

**Built:**
- `amt/data/dataset.py` -- new module holding:
  - `SongFeatures`, a frozen dataclass (`stem`, `cqt`, `frame_roll`,
    `onset_roll`) bundling one song's inputs and targets together.
  - `process_song(wav_path, midi_path, cfg) -> SongFeatures`, extracted
    directly from `scripts/validate_smd.py`'s inline loop body -- same
    load -> CQT -> sustain-pedal-extend -> piano-roll sequence, now
    called from one place instead of copy-pasted per script.
- `scripts/validate_smd.py` refactored to call `process_song()` instead
  of inlining the pipeline. Re-run against all 50 SMD pairs after the
  refactor: identical shapes to Friday's fixed run, confirming the
  extraction changed nothing about behavior.
- `scripts/cache_smd_features.py` -- runs `process_song()` over all 50
  SMD pairs and saves each song's `cqt`, `frame_roll`, `onset_roll`,
  and `stem` to a compressed `.npz` under `data/processed/smd/` (not
  committed -- derived data, covered by `.gitignore`'s `/data/` rule).

**Verified:**
- `validate_smd.py` post-refactor: 50/50 `ok`, shapes match pre-refactor
  exactly.
- `cache_smd_features.py`: 50/50 `ok`; `ls data/processed/smd | wc -l`
  confirms 50 files actually written to disk (not just 50 successful
  print statements); spot-checked one `.npz` by reloading it and
  confirming `cqt`/`frame_roll`/`onset_roll` shapes and `stem` match.

**Design note:** a plain Python string (`stem`) saved via
`np.savez_compressed` round-trips as a 0-d `numpy.ndarray`, not a
`str` -- needs `str(loaded["stem"])` to get a real string back. Minor
gotcha, not a problem for how `stem` is used here (filenames/bookkeeping
only), but worth remembering if metadata needs grow later.

**Not yet done:** song-level train/val/test splitting, and the
`mir_eval`-based evaluation module. These are next.

---

## Song-level splitting: group_durations + song_level_split

**Goal:** build a leak-free train/val/test split that groups whole
musical works together, rather than splitting by individual stem --
the same class of leak the original project's segment-level split had,
one level up.

**Built:**
- `group_durations()` and `song_level_split()`, added to
  `amt/data/dataset.py` alongside `SongFeatures`/`process_song()`.
  Given a stem->duration mapping and a stem->group mapping,
  `song_level_split()` shuffles whole *groups* (seeded via a dedicated
  `random.Random(cfg.split.seed)` instance, not the global `random`
  module, so an unrelated `random.*` call elsewhere can't disturb
  reproducibility) and greedily assigns them to train until
  `cfg.split.train_hours` is reached, then val until
  `cfg.split.val_hours`, with the remainder as test.
- `configs/smd_dev.yaml` -- small hour targets (`train_hours: 2.5`,
  `val_hours: 1.0`) for exercising the split on SMD locally. SMD's
  total ~4.7 hours is far below the MAESTRO-scale defaults
  (`train_hours: 45`); running the split with default config puts all
  of SMD into "train" with empty val/test -- not a bug, confirmed by
  testing it, just the wrong config for SMD's scale.
- `scripts/split_smd.py` -- builds per-stem durations cheaply via
  `librosa.get_duration(path=...)` (no full audio load needed) and
  runs the real split on SMD.

**Verified:**
- `pytest tests/` -- 22 passed, including two new tests in
  `tests/test_dataset.py`: one builds synthetic multi-stem groups and
  asserts every stem in a group lands in the same split (the actual
  leak-prevention guarantee), the other asserts the same seed produces
  an identical split across two calls.
- `scripts/split_smd.py` on real SMD data with `smd_dev.yaml`: 29
  train / 12 val / 9 test stems, hours close to the 2.5/1.0 targets,
  "No leaks found" confirmed by inspection.

**Known simplification, not fixed now (documented, not silent):**
SMD's stem->group rule is "each stem is its own group" -- it does
*not* group multi-movement sonatas (e.g. `Beethoven_Op027No1-01/-02/-03`)
together. A naive "split on first hyphen" alternative was tested; it
correctly grouped multi-movement sonatas but incorrectly merged
Chopin's Op. 28 (24 independent preludes) into one giant group.
Decided not worth solving precisely for SMD, since SMD's role here is
dev sandbox + later cross-dataset test, not where reported numbers
come from -- MAESTRO is, and MAESTRO already ships an official
work-level train/val/test split in its own metadata CSV, sidestepping
this ambiguity entirely. `song_level_split()` itself is written
generically (it consumes a group->stems mapping, not filename-parsing
logic), so it will be reused unchanged once MAESTRO's official split
column is wired in as the grouping source.

**Not yet done:** the `mir_eval`-based evaluation module. Next.

---