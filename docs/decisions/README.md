# Architecture Decision Records

Each file here is a short, dated record of one non-obvious engineering
decision: what problem it solves, what we chose, what we rejected, and
why. They are written as we make the decision, not reconstructed later.

This exists for three reasons:

1. **The capstone report writes itself.** "Why did you use pretty_midi
   instead of writing your own MIDI parser?" has an already-written
   answer with evidence, instead of a reconstructed memory.
2. **Evaluators and recruiters can see the reasoning, not just the
   result.** A bug fixed silently is invisible; a bug documented with
   its symptom, root cause, and fix is a demonstrated skill.
3. **Future us doesn't re-litigate settled questions**, or reintroduce
   a bug a past decision specifically avoided.

## Format

Each record is numbered sequentially and named `NNNN-short-title.md`,
with sections: Status, Context, Decision, Consequences. Keep them short
-- a paragraph or two per section is usually enough.

## Index

| # | Title |
|---|---|
| [0001](0001-pretty-midi-for-labels.md) | Use pretty_midi instead of hand-rolled MIDI parsing |
| [0002](0002-log-magnitude-cqt.md) | Log-magnitude CQT instead of the real part |
| [0003](0003-frame-rate-and-pitch-range.md) | 16 ms frames and a consistent 88-key pitch range |
