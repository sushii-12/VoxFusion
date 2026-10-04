import os
import random
import csv
from itertools import combinations
from collections import defaultdict

# Configuration
DATASET_ROOT = os.path.normpath(os.path.join("data", "LibriSpeech", "test-clean"))
OUTPUT_CSV = os.path.normpath(os.path.join("data", "ecapa_calibration_trials.csv"))
NUM_SAME = 1000
NUM_DIFF = 1000
SEED = 42

random.seed(SEED)

def get_audio_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".flac"):
                files.append(os.path.join(root, f))
    return files

def get_speaker_id(filepath):
    # LibriSpeech format: SPEAKERID-CHAPTERID-UTTID.flac
    basename = os.path.basename(filepath)
    speaker_id = basename.split("-")[0]
    return speaker_id

def normalize_path(filepath):
    # Convert to relative path from project root with forward slashes for consistency
    return filepath.replace("\\", "/")

def main():
    audio_files = get_audio_files(DATASET_ROOT)

    # Group by speaker
    speaker_to_files = defaultdict(list)
    for f in audio_files:
        spk = get_speaker_id(f)
        if not spk.isdigit():
            print(f"Warning: Invalid speaker ID found in {f}")
            continue
        speaker_to_files[spk].append(normalize_path(f))

    speakers = list(speaker_to_files.keys())

    # Generate same-speaker trials
    same_speaker_pool = set()
    for spk, files in speaker_to_files.items():
        if len(files) >= 2:
            pairs = list(combinations(files, 2))
            for p1, p2 in pairs:
                # normalize order to avoid duplicates
                if p1 > p2:
                    p1, p2 = p2, p1
                same_speaker_pool.add((p1, p2, spk, spk))

    same_speaker_trials = random.sample(list(same_speaker_pool), min(NUM_SAME, len(same_speaker_pool)))

    # Generate different-speaker trials
    diff_speaker_pool = set()
    attempts = 0
    max_attempts = NUM_DIFF * 10

    diff_speaker_trials = set()
    while len(diff_speaker_trials) < NUM_DIFF and attempts < max_attempts:
        attempts += 1
        spk1, spk2 = random.sample(speakers, 2)
        p1 = random.choice(speaker_to_files[spk1])
        p2 = random.choice(speaker_to_files[spk2])

        if p1 > p2:
            p1, p2 = p2, p1
            spk1, spk2 = spk2, spk1

        diff_speaker_trials.add((p1, p2, spk1, spk2))

    diff_speaker_trials = list(diff_speaker_trials)

    all_trials = []
    for p1, p2, s1, s2 in same_speaker_trials:
        all_trials.append({"label": 1, "audio1": p1, "audio2": p2, "speaker1": s1, "speaker2": s2})

    for p1, p2, s1, s2 in diff_speaker_trials:
        all_trials.append({"label": 0, "audio1": p1, "audio2": p2, "speaker1": s1, "speaker2": s2})

    # Shuffle the final list
    random.shuffle(all_trials)

    # Validation before writing
    assert len(all_trials) == NUM_SAME + NUM_DIFF, f"Expected {NUM_SAME+NUM_DIFF} trials, got {len(all_trials)}"
    assert sum(1 for t in all_trials if t["label"] == 1) == NUM_SAME
    assert sum(1 for t in all_trials if t["label"] == 0) == NUM_DIFF

    seen_pairs = set()
    missing_files = 0
    unique_speakers = set()

    for t in all_trials:
        p1 = t["audio1"]
        p2 = t["audio2"]
        s1 = t["speaker1"]
        s2 = t["speaker2"]

        if not os.path.exists(p1): missing_files += 1
        if not os.path.exists(p2): missing_files += 1

        if t["label"] == 1:
            assert s1 == s2
        else:
            assert s1 != s2

        # check duplicate pairs
        pair = (p1, p2) if p1 < p2 else (p2, p1)
        assert pair not in seen_pairs
        seen_pairs.add(pair)

        unique_speakers.add(s1)
        unique_speakers.add(s2)

    # Write to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "audio1", "audio2", "speaker1", "speaker2"])
        writer.writeheader()
        writer.writerows(all_trials)

    print(f"Total trials generated: {len(all_trials)}")
    print(f"Same-speaker trials: {sum(1 for t in all_trials if t['label'] == 1)}")
    print(f"Different-speaker trials: {sum(1 for t in all_trials if t['label'] == 0)}")
    print(f"Unique speakers represented: {len(unique_speakers)}")
    print(f"Missing files referenced: {missing_files}")
    print(f"Duplicate pairs: 0 (validated)")
    print(f"Output saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
