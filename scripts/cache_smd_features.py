"""
Cache CQT features, piano-roll labels, and continuous reference notes
for every SMD piece.

Audio comes from data/smd/wav_22050_mono -- the real recordings. (Do NOT
use midi_wav_22050_mono: that folder is audio synthesized from the MIDI,
which makes transcription unrealistically easy and inflates every score.)

Each .npz holds:
  cqt, frame_roll, onset_roll  -- model input / frame-level targets
  ref_onsets, ref_offsets      -- reference note times in seconds (float64),
  ref_pitches                     pedal-extended, NOT frame-quantized; note-level
                                  metrics are scored against these.
  stem

Usage:
    python3 scripts/cache_smd_features.py
"""
from pathlib import Path

import numpy as np

from amt.config import Config
from amt.data.dataset import process_song
from amt.data.midi import dedupe_notes, extend_notes_with_sustain_pedal, load_notes

cfg = Config()

midi_dir = Path("data/smd/midi")
wav_dir = Path("data/smd/wav_22050_mono")  # real recordings
out_dir = Path("data/processed/smd")
out_dir.mkdir(parents=True, exist_ok=True)

midi_stems = {p.stem for p in midi_dir.glob("*.mid")}
wav_stems = {p.stem for p in wav_dir.glob("*.wav")}
common_stems = midi_stems & wav_stems

print(f"{len(midi_stems)} midi, {len(wav_stems)} wav, {len(common_stems)} paired")


def reference_notes(midi_path: Path):
    if cfg.labels.extend_with_sustain_pedal:
        notes = extend_notes_with_sustain_pedal(
            str(midi_path), threshold=cfg.labels.sustain_pedal_threshold
        )
    else:
        notes = load_notes(str(midi_path))
    return dedupe_notes(notes)  # some SMD MIDI files list every note twice


results = []
for stem in sorted(common_stems):
    wav_path = wav_dir / f"{stem}.wav"
    midi_path = midi_dir / f"{stem}.mid"
    out_path = out_dir / f"{stem}.npz"
    try:
        song = process_song(wav_path, midi_path, cfg)
        notes = reference_notes(midi_path)
        np.savez_compressed(
            out_path,
            cqt=song.cqt,
            frame_roll=song.frame_roll,
            onset_roll=song.onset_roll,
            ref_onsets=np.array([n.start for n in notes], dtype=np.float64),
            ref_offsets=np.array([n.end for n in notes], dtype=np.float64),
            ref_pitches=np.array([n.pitch for n in notes], dtype=np.int64),
            stem=song.stem,
        )
        results.append((stem, "ok", song.cqt.shape, len(notes)))
    except Exception as e:
        results.append((stem, "FAILED", str(e)))

for r in results:
    print(r)

n_failed = sum(r[1] == "FAILED" for r in results)
print(f"\n{len(results) - n_failed} ok, {n_failed} failed")