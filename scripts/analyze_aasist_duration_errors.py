import csv
import soundfile as sf
from pathlib import Path
from collections import defaultdict


root = Path("data/aaspoof_sample/2021")
predictions = Path("results/aasist_evaluation/2021_predictions.csv")


# Group audio durations into bands
def duration_band(seconds):
    if seconds < 2:
        return "<2s"
    elif seconds < 4:
        return "2–4s"
    return "4s+"


# Store counts for each duration band and actual class
groups = defaultdict(lambda: {
    "total": 0,
    "errors": 0,
    "false_positives": 0,
    "false_negatives": 0,
})


# Read predictions and match them with audio durations
with predictions.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):

        # Skip files that failed during evaluation
        if row["status"] != "ok":
            continue

        audio_path = root / "audio" / row["filename"]

        info = sf.info(audio_path)
        duration = info.frames / info.samplerate

        band = duration_band(duration)
        actual = row["actual_label"]
        predicted = row["predicted_label"]

        key = (band, actual)
        stats = groups[key]

        stats["total"] += 1

        # A bona fide clip predicted as deepfake is a false positive.
        # A spoof clip predicted as bona fide is a false negative.
        is_error = (
            (actual == "bonafide" and predicted == "deepfake")
            or
            (actual == "spoof" and predicted == "bona_fide")
        )

        if is_error:
            stats["errors"] += 1

            if actual == "bonafide":
                stats["false_positives"] += 1

            elif actual == "spoof":
                stats["false_negatives"] += 1


# Print results
print("===== ASVspoof 2021 LA: Errors by Duration =====")
print(
    "Duration | Actual class | N   | Errors | "
    "Error rate | FP | FN"
)

total_fp = 0
total_fn = 0

for band in ("<2s", "2–4s", "4s+"):
    for label in ("bonafide", "spoof"):

        stats = groups[(band, label)]

        if stats["total"] == 0:
            continue

        error_rate = (
            100 * stats["errors"] / stats["total"]
        )

        total_fp += stats["false_positives"]
        total_fn += stats["false_negatives"]

        print(
            f"{band:8} | {label:11} | "
            f"{stats['total']:3} | "
            f"{stats['errors']:6} | "
            f"{error_rate:8.1f}% | "
            f"{stats['false_positives']:2} | "
            f"{stats['false_negatives']:2}"
        )


print("\n===== TOTAL ERRORS =====")
print(f"False positives: {total_fp}")
print(f"False negatives: {total_fn}")
print(f"Total errors:    {total_fp + total_fn}")