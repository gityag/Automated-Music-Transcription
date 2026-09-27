from __future__ import annotations

from amt.config import Config, SplitConfig
from amt.data.dataset import group_durations, song_level_split


def test_group_durations_sums_per_group():
    stem_to_duration = {"a1": 10.0, "a2": 20.0, "b1": 5.0}
    stem_to_group = {"a1": "A", "a2": "A", "b1": "B"}

    group_to_duration, group_to_stems = group_durations(stem_to_duration, stem_to_group)

    assert group_to_duration == {"A": 30.0, "B": 5.0}
    assert sorted(group_to_stems["A"]) == ["a1", "a2"]
    assert group_to_stems["B"] == ["b1"]


def _synthetic_groups(n_groups: int, stems_per_group: int, hours_per_group: float):
    """Build fake group_to_duration / group_to_stems for testing, without
    needing any real audio or MIDI files on disk."""
    group_to_duration = {}
    group_to_stems = {}
    for i in range(n_groups):
        group = f"group{i}"
        group_to_duration[group] = hours_per_group * 3600
        group_to_stems[group] = [f"{group}_stem{j}" for j in range(stems_per_group)]
    return group_to_duration, group_to_stems


def test_song_level_split_no_group_crosses_splits():
    group_to_duration, group_to_stems = _synthetic_groups(
        n_groups=10, stems_per_group=3, hours_per_group=1.0
    )
    cfg = Config(split=SplitConfig(train_hours=4.0, val_hours=2.0, seed=0))

    splits = song_level_split(group_to_duration, group_to_stems, cfg)

    stem_to_split = {}
    for split_name, stems in splits.items():
        for stem in stems:
            stem_to_split[stem] = split_name

    for group, stems in group_to_stems.items():
        splits_seen = {stem_to_split[stem] for stem in stems}
        assert len(splits_seen) == 1, f"{group}'s stems landed in multiple splits: {splits_seen}"

    total_stems = sum(len(stems) for stems in group_to_stems.values())
    assigned_stems = sum(len(stems) for stems in splits.values())
    assert assigned_stems == total_stems


def test_song_level_split_is_reproducible_with_same_seed():
    group_to_duration, group_to_stems = _synthetic_groups(
        n_groups=10, stems_per_group=2, hours_per_group=1.0
    )
    cfg = Config(split=SplitConfig(train_hours=4.0, val_hours=2.0, seed=42))

    splits_a = song_level_split(group_to_duration, group_to_stems, cfg)
    splits_b = song_level_split(group_to_duration, group_to_stems, cfg)

    assert splits_a == splits_b