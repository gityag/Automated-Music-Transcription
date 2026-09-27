from pathlib import Path

import numpy as np

from amt.config import Config
from amt.data.dataset import process_song

cfg = Config()

midi_dir = Path("data/smd/midi")
wav_dir = Path("data/smd/midi_wav_22050_mono")
out_dir = Path("data/processed/smd")
out_dir.mkdir(parents=True, exist_ok=True)

midi_stems = {p.stem for p in midi_dir.glob("*.mid")}
wav_stems = {p.stem for p in wav_dir.glob("*.wav")}
common_stems = midi_stems & wav_stems

print(f"{len(midi_stems)} midi, {len(wav_stems)} wav, {len(common_stems)} paired")

results = []
for stem in sorted(common_stems):
    wav_path = wav_dir / f"{stem}.wav"
    midi_path = midi_dir / f"{stem}.mid"
    out_path = out_dir / f"{stem}.npz"
    try:
        song = process_song(wav_path, midi_path, cfg)
        np.savez_compressed(
            out_path,
            cqt=song.cqt,
            frame_roll=song.frame_roll,
            onset_roll=song.onset_roll,
            stem=song.stem,
        )
        results.append((stem, "ok", song.cqt.shape))
    except Exception as e:
        results.append((stem, "FAILED", str(e)))

for r in results:
    print(r)