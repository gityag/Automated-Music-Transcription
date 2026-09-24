# 0001: Use pretty_midi instead of hand-rolled MIDI parsing

**Status:** Accepted

## Context

The original pipeline (`create_dataset.py`) parsed MIDI files by hand:
walking raw `mido` messages, accumulating delta-ticks, and converting to
seconds using a single tempo value read from the first track. Auditing
this code surfaced four concrete correctness bugs, not just style
issues:

1. **Single-tempo assumption.** `tempo_in_microsecs_per_beat` is read
   once from `midi_file.tracks[0]` and used for the whole song. A file
   with a tempo change partway through gets every note after that point
   at the wrong time.
2. **Re-struck notes merged.** `chop_simplified_midi` drops a `note_on`
   for any pitch already marked "on," so two fast repeated strikes of
   the same key become one long note. This is common in piano music
   (trills, repeated chords) and was silently discarding real events.
3. **False gap on held notes.** `add_note_onsets_to_beginning_when_needed`
   inserts a synthetic `note_off` at the segment boundary for a note that
   should keep sounding, then the encoder rounds that synthetic message
   to the nearest one-sixth-of-a-segment bucket -- so every note held
   across a segment boundary gets switched off for the last ~83 ms of
   the segment, on a regular grid, in every training example.
4. **`note_on velocity=0`, used by many MIDI writers as a note-off, was
   not special-cased anywhere in the ticks-to-seconds or on/off
   accounting**, so files that use this convention would silently miscount
   on/off pairs.

None of these are edge cases you can choose to ignore -- they fire on
ordinary piano recordings with normal playing technique.

## Decision

Parse MIDI with `pretty_midi`, which resolves the full tempo map,
merges `note_on velocity=0` into note-off, and returns notes as
`(pitch, start_seconds, end_seconds, velocity)` directly. This removes
~300 lines of tick/tempo bookkeeping and the four bugs above along with
it, because none of that logic exists anymore.

The one piece of musical logic pretty_midi does *not* decide for us --
whether and how the sustain pedal extends a note's sounding duration --
is implemented explicitly in `amt.data.midi.extend_notes_with_sustain_pedal`,
using the standard rule from the Onsets and Frames literature: extend a
note's end to the pedal release time, but never past the next onset of
the same pitch. See `tests/test_midi.py` for this rule expressed as
executable cases, including the re-struck-note and held-note-at-boundary
scenarios above -- both now behave correctly by construction, and the
tests exist so they stay that way.

## Consequences

- One new dependency (`pretty_midi`), which is standard in this field
  and already a dependency of `mir_eval`-adjacent tooling.
- The sustain-pedal rule is a single, tested function, so it is applied
  identically to ground-truth labels and to decoded model output --
  previously there was no equivalent function on the decode side at all
  (`decode_midi.py`'s `reconstruct_midi` never actually wrote a file,
  a separate bug).
- Note-level metrics computed on the *old* labels are not comparable to
  metrics computed on the new ones; any "before" baseline must be
  re-scored under the corrected pipeline, not compared to old numbers
  from the legacy code.
