from pathlib import Path
import sys
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_detection import AASISTDetector

df = pd.read_csv(PROJECT_ROOT / "data/evaluation_manifest.csv")

detector = AASISTDetector()
results = []

for _, row in df.iterrows():
    path = PROJECT_ROOT / row["path"]

    print(f"Processing: {path.name}")

    prediction = detector.predict(str(path))

    results.append({
        "label": row["label"],
        "path": str(path),
        "result": prediction
    })

(PROJECT_ROOT / "results").mkdir(exist_ok=True)

with open(PROJECT_ROOT / "results/central_aasist_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nAASIST baseline completed: {len(results)} samples")
