from pathlib import Path
import librosa
from amt.config import Config
from amt.features.cqt import compute_log_cqt, frame_rate
from amt.data.midi import extend_notes_with_sustain_pedal, notes_to_piano_roll

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
        # reuse the exact steps from your one-file test above:
        # load audio, compute duration and fs, get notes, build
        # frame_roll/onset_roll, compute cqt
        sr_target = cfg.audio.sample_rate
        audio, sr = librosa.load(wav_path, sr=sr_target)
        fs = frame_rate(sr,  cfg.cqt.hop_length) # one line, using the function you found in cqt.py
        cqt = compute_log_cqt(audio, sr, cfg.cqt.fmin_midi, cfg.cqt.n_bins, cfg.cqt.bins_per_octave, cfg.cqt.hop_length)
        notes = extend_notes_with_sustain_pedal(midi_path, threshold=cfg.labels.sustain_pedal_threshold)
        frame_roll, onset_roll = notes_to_piano_roll(notes, n_frames=cqt.shape[1], fs=fs, pitch_low=cfg.labels.pitch_low, pitch_high=cfg.labels.pitch_high)
        results.append((stem, "ok", cqt.shape, frame_roll.shape))
    except Exception as e:
        results.append((stem, "FAILED", str(e)))

for r in results:
    print(r)