import csv
import time
from pathlib import Path

from src.deepfake_detection import AASISTDetector


ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = ROOT / "data" / "aaspoof_gsm" / "2021"
LABELS_PATH = ROOT / "data" / "aaspoof_sample" / "2021" / "labels.csv"
OUTPUT_PATH = ROOT / "results" / "aasist_gsm_2021_predictions.csv"


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


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LABELS_PATH.open(
        newline="", encoding="utf-8"
    ) as file:
        labels = list(csv.DictReader(file))

    if not labels:
        raise ValueError(f"No labels found: {LABELS_PATH}")

    print("Loading AASIST checkpoint once...")
    detector = AASISTDetector()

    results = []
    errors = []
    start = time.time()

    print("\n" + "=" * 55)
    print(f"Evaluating GSM ASVspoof 2021 LA ({len(labels)} clips)")
    print("=" * 55)

    for index, item in enumerate(labels, start=1):
        filename = item["filename"]
        actual_label = item["label"].strip().lower()

        if actual_label not in LABEL_MAP:
            raise ValueError(
                f"Unknown label {actual_label!r} in {LABELS_PATH}"
            )

        audio_path = AUDIO_DIR / filename

        try:
            prediction = detector.predict(audio_path)
            predicted_label = prediction["prediction"]

            row = {
                "filename": filename,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "bona_fide_score": prediction["bona_fide_score"],
                "deepfake_score": prediction["deepfake_score"],
                "raw_bona_fide_score": prediction["raw_bona_fide_score"],
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
        row for row in results
        if row["status"] == "ok"
    ]

    metrics = calculate_metrics(successful)

    metrics["dataset"] = "ASVspoof 2021 LA - GSM"
    metrics["failed_files"] = len(errors)
    metrics["elapsed_seconds"] = round(time.time() - start, 2)

    with OUTPUT_PATH.open(
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

    print(f"\nResults: {OUTPUT_PATH}")
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


if __name__ == "__main__":
    main()