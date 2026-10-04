import os
import sys
import csv
import json
import numpy as np
import torch
from sklearn.metrics import roc_curve

# Add src to pythonpath so we can import src.speaker_verification
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.speaker_verification import SpeakerVerifier

def main():
    trials_csv = "data/ecapa_calibration_trials.csv"
    output_csv = "results/ecapa_calibration_trials.csv"
    output_json = "results/ecapa_calibration_summary.json"

    os.makedirs("results", exist_ok=True)

    # Read trials
    trials = []
    with open(trials_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['label'] = int(row['label'])
            trials.append(row)

    print(f"Loaded {len(trials)} trials.")

    # Collect unique audio files
    unique_audios = set()
    for row in trials:
        unique_audios.add(row['audio1'])
        unique_audios.add(row['audio2'])

    print(f"Extracting embeddings for {len(unique_audios)} unique audio files...")

    verifier = SpeakerVerifier()
    embeddings_cache = {}

    for audio_path in unique_audios:
        try:
            # Use the existing implementation to load and encode
            audio_tensor = verifier._load_audio(audio_path)
            emb = verifier._encode_audio(audio_tensor)
            embeddings_cache[audio_path] = emb
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            sys.exit(1)

    print("Computing similarities...")

    for row in trials:
        emb1 = embeddings_cache[row['audio1']]
        emb2 = embeddings_cache[row['audio2']]

        # Compute cosine similarity using the model's similarity method
        score = verifier.model.similarity(emb1.to(verifier.device), emb2.to(verifier.device))
        score_value = float(score.squeeze().item())
        row['cosine_similarity'] = score_value

    # Write output CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["label", "audio1", "audio2", "speaker1", "speaker2", "cosine_similarity"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    # Calculate metrics
    y_true = np.array([t['label'] for t in trials])
    y_score = np.array([t['cosine_similarity'] for t in trials])

    # Higher score means more likely same speaker, so ROC curve uses y_score directly
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    # Find EER where FAR (fpr) and FRR (fnr) are closest
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Evaluate at current threshold 0.25
    current_threshold = 0.25
    preds = y_score > current_threshold

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    far_025 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr_025 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy_025 = (tp + tn) / (tp + tn + fp + fn)
    precision_025 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_025 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_025 = 2 * (precision_025 * recall_025) / (precision_025 + recall_025) if (precision_025 + recall_025) > 0 else 0.0

    summary = {
        "total_trials": len(trials),
        "same_speaker_trials": int(np.sum(y_true == 1)),
        "different_speaker_trials": int(np.sum(y_true == 0)),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "current_threshold": current_threshold,
        "current_threshold_far": float(far_025),
        "current_threshold_frr": float(frr_025),
        "current_threshold_accuracy": float(accuracy_025),
        "current_threshold_precision": float(precision_025),
        "current_threshold_recall": float(recall_025),
        "current_threshold_f1": float(f1_025),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n========== CALIBRATION REPORT ==========")
    print(f"Unique embeddings computed: {len(unique_audios)}")
    print(f"Total trials:               {summary['total_trials']}")
    print(f"EER:                        {summary['eer']:.4f}")
    print(f"EER Threshold:              {summary['eer_threshold']:.4f}")
    print(f"Current Threshold:          {summary['current_threshold']}")
    print(f"FAR at 0.25:                {summary['current_threshold_far']:.4f}")
    print(f"FRR at 0.25:                {summary['current_threshold_frr']:.4f}")
    print(f"Accuracy at 0.25:           {summary['current_threshold_accuracy']:.4f}")
    print(f"F1 at 0.25:                 {summary['current_threshold_f1']:.4f}")
    print(f"TP / TN / FP / FN:          {tp} / {tn} / {fp} / {fn}")
    print("========================================")

if __name__ == "__main__":
    main()
