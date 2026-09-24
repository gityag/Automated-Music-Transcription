"""
Constant-Q Transform feature extraction.

Replaces the legacy `audio_segments_cqt`, which kept only the real part
of the (complex) CQT. That value oscillates with the phase of the
waveform at the window boundary, so the same note at the same loudness
produced different features depending on exactly when the analysis
window started -- effectively injecting label-independent noise into
every training example. See docs/decisions/0002-log-magnitude-cqt.md.

This module uses log-magnitude instead: magnitude is phase-invariant,
and the log compression matches the ear's roughly logarithmic loudness
perception, which is standard for audio-to-pitch tasks.
"""
from __future__ import annotations

import librosa
import numpy as np


def compute_log_cqt(
    audio: np.ndarray,
    sr: int,
    fmin_midi: int,
    n_bins: int,
    bins_per_octave: int,
    hop_length: int,
) -> np.ndarray:
    """Compute a log-magnitude CQT.

    Returns an array of shape (n_bins, n_frames), dtype float32. Frame i
    is centered at time `i * hop_length / sr` seconds, the same
    convention `amt.data.midi.notes_to_piano_roll` uses for `fs =
    sr / hop_length` -- keep these two in lockstep, or features and
    targets drift apart on the time axis.
    """
    fmin = librosa.midi_to_hz(fmin_midi)
    c = librosa.cqt(
        y=audio,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
    )
    magnitude = np.abs(c)
    # ref=1.0 (an absolute reference), not ref=np.max (the clip's own
    # loudest bin). A per-clip max reference means the same note at the
    # same physical loudness gets a different dB value depending on what
    # else happens to be in that particular audio clip -- undesirable
    # for a feature that's meant to generalize across clips of different
    # length and content. See docs/decisions/0002-log-magnitude-cqt.md.
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=1.0)
    return log_magnitude.astype(np.float32)


def frame_rate(sr: int, hop_length: int) -> float:
    """Frames per second produced by `compute_log_cqt` for this sr/hop."""
    return sr / hop_length
