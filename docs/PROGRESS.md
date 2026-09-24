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
