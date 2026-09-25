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