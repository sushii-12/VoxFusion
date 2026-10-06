from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aasist"))

from models.AASIST import Model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600
BATCH_SIZE = 4

VAL_CSV = ROOT / "data" / "aasist_finetune" / "val.csv"

ORIGINAL = ROOT / "aasist" / "models" / "weights" / "AASIST.pth"
FINETUNED = ROOT / "models" / "AASIST_gsm_finetuned_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": NUM_SAMPLES,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64]
    ],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}


# ---------------------------------------------------------
# Audio
# ---------------------------------------------------------

def load_audio(path):

    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != SAMPLE_RATE:
        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sr
        )

    audio = audio.astype(np.float32)

    if len(audio) < NUM_SAMPLES:
        repeats = int(np.ceil(NUM_SAMPLES / len(audio)))
        audio = np.tile(audio, repeats)

    if len(audio) > NUM_SAMPLES:
        audio = audio[:NUM_SAMPLES]

    return torch.from_numpy(audio)


# ---------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------

class ValidationDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        rows = []

        for _, row in self.df.iterrows():

            rows.append({
                "path": row["clean"],
                "label": int(row["label"]),
                "codec": "clean"
            })

            rows.append({
                "path": row["gsm"],
                "label": int(row["label"]),
                "codec": "gsm"
            })

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        audio = load_audio(row["path"])

        return (
            audio,
            row["label"],
            row["codec"],
            row["path"]
        )


dataset = ValidationDataset(VAL_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------------
# Model loader
# ---------------------------------------------------------

def load_model(checkpoint):

    model = Model(MODEL_CONFIG)

    state_dict = torch.load(
        checkpoint,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model):

    results = []

    with torch.no_grad():

        for audio, labels, codecs, paths in loader:

            audio = audio.to(DEVICE)

            _, output = model(audio)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            # Official AASIST:
            # class 1 = bona fide
            # class 0 = spoof
            deepfake_scores = probabilities[:, 0]

            predictions = torch.argmax(
                output,
                dim=1
            )

            for i in range(len(labels)):

                results.append({
                    "path": paths[i],
                    "codec": codecs[i],
                    "label": int(labels[i]),
                    "prediction": int(predictions[i].cpu()),
                    "deepfake_score": float(
                        deepfake_scores[i].cpu()
                    )
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def metrics(df):

    y_true = df["label"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    tp = int(((y_true == 0) & (y_pred == 0)).sum())
    tn = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 1) & (y_pred == 0)).sum())
    fn = int(((y_true == 0) & (y_pred == 1)).sum())

    accuracy = (tp + tn) / len(df)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

print("=" * 60)
print("AASIST ORIGINAL vs GSM-FINETUNED")
print("=" * 60)

print("Device:", DEVICE)

print("\nLoading original model...")
original_model = load_model(ORIGINAL)

print("Loading fine-tuned model...")
finetuned_model = load_model(FINETUNED)


print("\nEvaluating original...")
original_results = evaluate(original_model)

print("Evaluating fine-tuned...")
finetuned_results = evaluate(finetuned_model)


# ---------------------------------------------------------
# Overall + codec-specific metrics
# ---------------------------------------------------------

for name, results in [
    ("ORIGINAL AASIST", original_results),
    ("FINE-TUNED AASIST", finetuned_results)
]:

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    for codec in ["clean", "gsm"]:

        subset = results[
            results["codec"] == codec
        ]

        m = metrics(subset)

        print(f"\n{codec.upper()}")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1:        {m['f1']:.4f}")

        print(
            f"Confusion: "
            f"TN={m['tn']} "
            f"FP={m['fp']} "
            f"FN={m['fn']} "
            f"TP={m['tp']}"
        )


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

output_dir = ROOT / "results" / "aasist_finetuning"
output_dir.mkdir(parents=True, exist_ok=True)

original_results.to_csv(
    output_dir / "original_validation.csv",
    index=False
)

finetuned_results.to_csv(
    output_dir / "finetuned_validation.csv",
    index=False
)

print("\nResults saved to:")
print(output_dir)