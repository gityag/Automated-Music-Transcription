from pathlib import Path
from amt.config import Config
from amt.data.dataset import process_song

cfg = Config()

midi_dir = Path("data/smd/midi")
wav_dir = Path("data/smd/midi_wav_22050_mono")

midi_stems = {p.stem for p in midi_dir.glob("*.mid")}   # what extension?
wav_stems = {p.stem for p in wav_dir.glob("*.wav")}     # what extension?

common_stems = midi_stems & wav_stems 

print(f"{len(midi_stems)} midi, {len(wav_stems)} wav, {len(common_stems)} paired")

results = []
for stem in sorted(common_stems):
    wav_path = wav_dir / f"{stem}.wav"
    midi_path = midi_dir / f"{stem}.mid"
    try:
        song = process_song(wav_path, midi_path, cfg)
        results.append((stem, "ok", song.cqt.shape, song.frame_roll.shape))
    except Exception as e:
        results.append((stem, "FAILED", str(e)))

for r in results:
    print(r)