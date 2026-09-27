from pathlib import Path
import librosa
from amt.config import Config
from amt.data.dataset import group_durations, song_level_split

cfg = Config.from_yaml("configs/smd_dev.yaml")
midi_dir = Path("data/smd/midi")
wav_dir = Path("data/smd/midi_wav_22050_mono")

midi_stems = {p.stem for p in midi_dir.glob("*.mid")}
wav_stems = {p.stem for p in wav_dir.glob("*.wav")}
common_stems = sorted(midi_stems & wav_stems)

stem_to_duration = {stem: librosa.get_duration(path=wav_dir / f"{stem}.wav") for stem in common_stems}
stem_to_group = {stem: stem for stem in common_stems}

group_to_duration, group_to_stems = group_durations(stem_to_duration, stem_to_group)
splits = song_level_split(group_to_duration, group_to_stems, cfg)

for name in ("train", "val", "test"):
    total_hours = sum(stem_to_duration[s] for s in splits[name]) / 3600
    print(name, len(splits[name]), "stems,", round(total_hours, 2), "hours")

all_stems_seen = set()
overlap_found = False
for name in ("train", "val", "test"):
    for stem in splits[name]:
        if stem in all_stems_seen:
            print("LEAK:", stem)
            overlap_found = True
        all_stems_seen.add(stem)
print("No leaks found" if not overlap_found else "LEAKS FOUND")