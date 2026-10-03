import json
import glob
import os

import zstandard as zstd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


HF_BASE = os.path.expanduser(
    r"~\.cache\huggingface\hub\datasets--rishia2220--icfd-31k"
)

SNAPSHOT = os.path.join(
    HF_BASE,
    "snapshots",
    "e91937a718b4cec97b8c06820225a9bdc30dd84a",
    "source_conversations",
)

MODEL_DIR = os.path.join("models", "scam_intent_v2")


def load_validation_data():
    texts = []
    labels = []

    files = sorted(
        glob.glob(
            os.path.join(
                SNAPSHOT,
                "validation-*-of-00002.jsonl.zst"
            )
        )
    )

    if len(files) != 2:
        raise RuntimeError(
            f"Expected 2 validation shards, found {len(files)}"
        )

    for path in files:
        print(f"Loading: {os.path.basename(path)}")

        with open(path, "rb") as f:
            reader = zstd.ZstdDecompressor().stream_reader(f)
            data = reader.read().decode("utf-8")

        for line in data.splitlines():
            if not line.strip():
                continue

            record = json.loads(line)

            verdict = record.get("final_verdict")

            if verdict not in {"YES", "NO"}:
                continue

            transcript = record.get("transcript", [])

            conversation_text = " ".join(
                turn.get("text", "")
                for turn in transcript
                if turn.get("text")
            ).strip()

            if not conversation_text:
                continue

            texts.append(conversation_text)
            labels.append(1 if verdict == "YES" else 0)

    return texts, labels


def main():
    print("Loading trained V2 model...")

    vectorizer = load(
        os.path.join(
            MODEL_DIR,
            "tfidf_vectorizer.joblib"
        )
    )

    classifier = load(
        os.path.join(
            MODEL_DIR,
            "logistic_regression.joblib"
        )
    )

    print("Loading ICFD-31k validation data...")

    texts, y_true = load_validation_data()

    print(f"Validation conversations: {len(texts)}")
    print(f"Actual scams: {sum(y_true)}")
    print(f"Actual non-scams: {len(y_true) - sum(y_true)}")

    print("\nGenerating predictions...")

    X = vectorizer.transform(texts)

    y_pred = classifier.predict(X)

    print("\n===== V2 VALIDATION RESULTS =====")

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"Recall   : "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(
        f"F1       : "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    print("\nConfusion Matrix:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Non-scam", "Scam"],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
