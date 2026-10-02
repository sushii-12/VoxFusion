import csv
import sys
import time
from pathlib import Path

from src.deepfake_detection import AASISTDetector


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "aaspoof_sample"
OUTPUT_ROOT = ROOT / "results" / "aasist_evaluation"

YEARS = ["2019", "2021"]

# Positive class: spoof / deepfake
LABEL_MAP = {
    "bonafide": "bona_fide",
    "spoof": "deepfake",
}


def calculate_metrics(rows):
    tp = fp = tn = fn = 0

    for row in rows:
        actual = row["actual_label"]
        predicted = row["predicted_label"]

        if actual == "spoof" and predicted == "deepfake":
            tp += 1
        elif actual == "bonafide" and predicted == "deepfake":
            fp += 1
        elif actual == "bonafide" and predicted == "bona_fide":
            tn += 1
        elif actual == "spoof" and predicted == "bona_fide":
            fn += 1

    total = tp + fp + tn + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "total_evaluated": total,
        "TP_spoof_detected": tp,
        "FP_bonafide_flagged": fp,
        "TN_bonafide_correct": tn,
        "FN_spoof_missed": fn,
        "accuracy": accuracy,
        "precision_spoof": precision,
        "recall_spoof": recall,
        "f1_spoof": f1,
        "false_negative_rate": fnr,
    }


def evaluate_year(detector, year):
    year_dir = DATA_ROOT / year
    audio_dir = year_dir / "audio"
    labels_path = year_dir / "labels.csv"

    with labels_path.open(
        newline="", encoding="utf-8"
    ) as file:
        labels = list(csv.DictReader(file))

    if not labels:
        raise ValueError(f"No labels found: {labels_path}")

    results = []
    errors = []
    start = time.time()

    print(f"\n{'=' * 55}")
    print(f"Evaluating ASVspoof {year} LA ({len(labels)} clips)")
    print("=" * 55)

    for index, item in enumerate(labels, start=1):
        filename = item["filename"]
        actual_label = item["label"].strip().lower()

        if actual_label not in LABEL_MAP:
            raise ValueError(
                f"Unknown label {actual_label!r} in {labels_path}"
            )

        audio_path = audio_dir / filename

        try:
            prediction = detector.predict(audio_path)
            predicted_label = prediction["prediction"]

            row = {
                "filename": filename,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "bona_fide_score": prediction["bona_fide_score"],
                "deepfake_score": prediction["deepfake_score"],
                "raw_bona_fide_score": prediction[
                    "raw_bona_fide_score"
                ],
                "status": "ok",
                "error": "",
            }
            results.append(row)

        except Exception as exc:
            row = {
                "filename": filename,
                "actual_label": actual_label,
                "predicted_label": "",
                "bona_fide_score": "",
                "deepfake_score": "",
                "raw_bona_fide_score": "",
                "status": "error",
                "error": repr(exc),
            }
            results.append(row)
            errors.append(row)

        if index % 10 == 0 or index == len(labels):
            print(f"Processed {index}/{len(labels)}")

    successful = [
        row for row in results if row["status"] == "ok"
    ]
    metrics = calculate_metrics(successful)

    metrics["dataset"] = f"ASVspoof {year} LA"
    metrics["failed_files"] = len(errors)
    metrics["elapsed_seconds"] = round(time.time() - start, 2)

    predictions_path = OUTPUT_ROOT / f"{year}_predictions.csv"
    with predictions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
            "filename",
            "actual_label",
            "predicted_label",
            "bona_fide_score",
            "deepfake_score",
            "raw_bona_fide_score",
            "status",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults: {predictions_path}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(errors)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_spoof']:.4f}")
    print(f"Recall:    {metrics['recall_spoof']:.4f}")
    print(f"F1-score:  {metrics['f1_spoof']:.4f}")
    print(f"FNR:       {metrics['false_negative_rate']:.4f}")
    print(
        "Confusion matrix counts (spoof positive): "
        f"TP={metrics['TP_spoof_detected']}, "
        f"FP={metrics['FP_bonafide_flagged']}, "
        f"TN={metrics['TN_bonafide_correct']}, "
        f"FN={metrics['FN_spoof_missed']}"
    )

    if errors:
        print("\nFirst errors:")
        for error in errors[:5]:
            print(error["filename"], error["error"])

    return metrics


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading AASIST checkpoint once...")
    detector = AASISTDetector()

    summaries = []
    for year in YEARS:
        summaries.append(evaluate_year(detector, year))

    summary_path = OUTPUT_ROOT / "summary.csv"
    fields = list(summaries[0].keys())

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nEvaluation complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
