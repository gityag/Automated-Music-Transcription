# 0003: 16 ms frames and a consistent 88-key pitch range

**Status:** Accepted

## Context

Two related issues in the legacy pipeline:

**Time resolution.** `encode_midi_segment` discretized each 0.5 s MIDI
segment into 6 bins, i.e. ~83 ms per bin. `mir_eval`'s standard onset
tolerance for transcription evaluation is ±50 ms. A quantization step
larger than the tolerance it will be judged against caps the achievable
note-onset F1 before the model does anything -- some correctly-detected
onsets are guaranteed to land in the wrong bin purely from rounding.

**Pitch range mismatch.** `encode_midi_segments.py` hardcodes the label
range to MIDI 21-107 (87 notes, dropping C8/MIDI 108). Separately,
librosa's CQT with default settings starting at C1 covers MIDI 24-107.
The *input* features and the *output* targets were silently covering
different pitch ranges (24-107 vs. 21-107) -- meaning A0-B0 had no
usable input, and the model was never even asked about roughly a third
of a semitone range it might see in real recordings.

## Decision

- Use `hop_length=256` at `sample_rate=16000`, giving a 16 ms frame
  step (an analysis window of about twice that, ~32 ms). This is
  comfortably inside the 50 ms onset tolerance and matches common
  practice in the transcription literature (e.g. Onsets and Frames
  uses similar frame rates).
- Fix the pitch range at the full 88-key piano, MIDI 21-108, for both
  the CQT input (`fmin_midi=21, n_bins=88`) and the label piano roll
  (`pitch_low=21, pitch_high=108`).
- Enforce the two stay consistent in code, not just in comments:
  `Config.__post_init__` raises `ValueError` if `cqt.fmin_midi !=
  labels.pitch_low` or if `cqt.n_bins != pitch_high - pitch_low + 1`.
  See `tests/test_config.py` for both mismatch cases as regression
  tests. This turns "the input and target silently disagree" from a
  bug that only shows up as unexplained poor accuracy into a
  `ValueError` at config-construction time.

## Consequences

- Six times more label frames per second than the legacy encoding
  (~83 ms -> ~16 ms bins), which is a proportionally larger label
  array; acceptable at the dataset sizes this project uses (see the
  project roadmap's storage estimate).
- Any future change to sample rate, hop length, or pitch range must go
  through `Config`, where the validation lives, rather than being
  edited independently in a features module and a labels module the
  way the legacy scripts allowed.
