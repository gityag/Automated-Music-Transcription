# 0002: Log-magnitude CQT instead of the real part

**Status:** Accepted

## Context

`cqt.py` in the legacy pipeline computed a Constant-Q Transform and kept
only `.real`, discarding the imaginary component. The CQT is a complex-
valued transform; its magnitude, `sqrt(real^2 + imag^2)`, describes how
much energy is present at a given pitch, while the real/imaginary split
describes the *phase* of that energy relative to an arbitrary reference
point in time (the start of the analysis window).

Phase is not musically meaningful here -- the same note played at the
same loudness starting 5 ms earlier or later produces a completely
different real-part value, purely because the window boundary lands at
a different point in the waveform's cycle. `tests/test_cqt.py::test_phase_invariance`
demonstrates this directly: two identical 440 Hz tones differing only in
starting phase feed the model two different-looking inputs for the
`.real` feature. In effect, the original feature had a large,
label-independent noise source baked into every training example.

## Decision

Use log-magnitude instead: `librosa.amplitude_to_db(np.abs(cqt), ref=1.0)`.
Magnitude is phase-invariant by construction, and log compression matches
the (roughly logarithmic) way loudness is perceived and is standard
practice for audio-to-pitch tasks.

One further detail, found while writing the phase-invariance test: the
first implementation used `ref=np.max`, librosa's usual convention for
plotting. That normalizes every frame in a clip against *that clip's own
loudest frame*. For a feature meant to be comparable across many
different clips, this is itself a bug: the same physical note at the
same loudness gets a different dB value depending on what else happens
to be in that particular training example. `ref=1.0` gives an absolute
reference instead, so a note's feature value depends only on the note,
not on its neighbors in the clip.

## Consequences

- One-line implementation change from the legacy code, but it is likely
  the single largest contributor to the accuracy gap between the
  original model and the corrected pipeline -- this is exactly the kind
  of claim the ablation study (see the project roadmap) is designed to
  quantify rather than assert.
- Log-magnitude values are unbounded below (silence -> very negative
  dB); the training pipeline should clip or otherwise bound this range
  before it reaches the model, which is a detail deferred to the
  dataset/normalization step, not this module.
