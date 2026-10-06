from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
AASIST_ROOT = ROOT / "aasist"

sys.path.insert(0, str(AASIST_ROOT))

from models.AASIST import Model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600

BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

SEED = 42

TRAIN_CSV = ROOT / "data" / "aasist_finetune" / "train.csv"
VAL_CSV = ROOT / "data" / "aasist_finetune" / "val.csv"

CHECKPOINT = AASIST_ROOT / "models" / "weights" / "AASIST.pth"

OUTPUT_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "AASIST_gsm_finetuned_best.pth"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("VoxFusion AASIST GSM Fine-Tuning")
print("=" * 60)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print()


# ---------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio: {path}")

    # Resample to 16 kHz
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr)

    audio = audio.astype(np.float32)

    # Match official AASIST input length
    if len(audio) < NUM_SAMPLES:
        repeat_count = int(np.ceil(NUM_SAMPLES / len(audio)))
        audio = np.tile(audio, repeat_count)

    # Random crop during training
    if len(audio) > NUM_SAMPLES:
        start = np.random.randint(0, len(audio) - NUM_SAMPLES + 1)
        audio = audio[start:start + NUM_SAMPLES]

    return audio


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class AASISTFineTuneDataset(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty: {csv_path}")

        required = {"clean", "gsm", "label"}

        if not required.issubset(self.df.columns):
            raise ValueError(
                f"{csv_path} must contain columns: {required}"
            )

    def __len__(self):
        # Each original recording produces:
        # 1 clean example + 1 GSM example
        return len(self.df) * 2

    def __getitem__(self, index):

        row_index = index // 2
        is_gsm = index % 2 == 1

        row = self.df.iloc[row_index]

        path = row["gsm"] if is_gsm else row["clean"]
        label = int(row["label"])

        audio = load_audio(path)

        audio = torch.from_numpy(audio)

        return audio, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

train_dataset = AASISTFineTuneDataset(TRAIN_CSV)
val_dataset = AASISTFineTuneDataset(VAL_CSV)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))
print()


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------

model_config = {
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
# Create model
# ---------------------------------------------------------

print("Creating AASIST model...")

model = Model(model_config)

state_dict = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)

print("Loaded pretrained checkpoint:", CHECKPOINT)
print()


# ---------------------------------------------------------
# Loss / optimizer
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(labels, predictions):

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    accuracy = (tp + tn) / max(len(labels), 1)

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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

best_val_f1 = -1.0

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0.0
    train_labels = []
    train_predictions = []

    for batch_idx, (audio, labels) in enumerate(train_loader):

        audio = audio.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        _, output = model(audio)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            output,
            dim=1
        )

        train_labels.extend(
            labels.detach().cpu().numpy()
        )

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        if (batch_idx + 1) % 10 == 0:
            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_metrics = calculate_metrics(
        train_labels,
        train_predictions
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_labels = []
    val_predictions = []

    with torch.no_grad():

        for audio, labels in val_loader:

            audio = audio.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            _, output = model(audio)

            loss = criterion(output, labels)

            val_loss += loss.item()

            predictions = torch.argmax(
                output,
                dim=1
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

    val_metrics = calculate_metrics(
        val_labels,
        val_predictions
    )

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print()
    print("-" * 60)
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train F1:   {train_metrics['f1']:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_metrics['accuracy']:.4f}")
    print(f"Val Prec:   {val_metrics['precision']:.4f}")
    print(f"Val Recall: {val_metrics['recall']:.4f}")
    print(f"Val F1:     {val_metrics['f1']:.4f}")
    print(
        f"Val CM:     "
        f"TN={val_metrics['tn']} "
        f"FP={val_metrics['fp']} "
        f"FN={val_metrics['fn']} "
        f"TP={val_metrics['tp']}"
    )
    print("-" * 60)

    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_metrics["f1"] > best_val_f1:

        best_val_f1 = val_metrics["f1"]

        torch.save(
            model.state_dict(),
            BEST_CHECKPOINT
        )

        print(
            f"✓ Saved best checkpoint "
            f"(Val F1={best_val_f1:.4f})"
        )

    print()


print("=" * 60)
print("Fine-tuning complete")
print("Best checkpoint:", BEST_CHECKPOINT)
print("Best validation F1:", best_val_f1)
print("=" * 60)