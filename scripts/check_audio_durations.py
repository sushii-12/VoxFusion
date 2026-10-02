import csv
import soundfile as sf
from pathlib import Path
from collections import defaultdict

root = Path("data/aaspoof_sample")

for year in ("2019", "2021"):
    groups = defaultdict(list)
    labels_path = root / year / "labels.csv"

    with labels_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = root / year / "audio" / row["filename"]
            info = sf.info(path)
            duration = info.frames / info.samplerate
            groups[row["label"]].append(duration)

    print(f"\n===== ASVspoof {year} LA =====")
    for label, durations in sorted(groups.items()):
        durations.sort()
        n = len(durations)
        median = durations[n // 2] if n % 2 else (
            durations[n // 2 - 1] + durations[n // 2]
        ) / 2

        print(
            f"{label}: n={n}, "
            f"min={durations[0]:.2f}s, "
            f"median={median:.2f}s, "
            f"max={durations[-1]:.2f}s"
        )
