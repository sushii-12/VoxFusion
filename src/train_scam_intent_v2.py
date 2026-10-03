import json
import glob
import os
import zstandard as zstd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump


HF_BASE = os.path.expanduser(
    r"~\.cache\huggingface\hub\datasets--rishia2220--icfd-31k"
)

SNAPSHOT = os.path.join(
    HF_BASE,
    "snapshots",
    "e91937a718b4cec97b8c06820225a9bdc30dd84a",
    "source_conversations",
)

OUTPUT_DIR = os.path.join("models", "scam_intent_v2")


def load_conversations():
    texts = []
    labels = []

    files = sorted(
        glob.glob(
            os.path.join(
                SNAPSHOT,
                "train-*-of-00010.jsonl.zst"
            )
        )
    )

    if len(files) != 10:
        raise RuntimeError(
            f"Expected 10 training shards, found {len(files)}"
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

            transcript = record.get("transcript", [])
            verdict = record.get("final_verdict")

            if verdict not in {"YES", "NO"}:
                continue

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
    print("Loading ICFD-31k training data...")
    texts, labels = load_conversations()

    print(f"Training conversations: {len(texts)}")
    print(f"Scam conversations: {sum(labels)}")
    print(f"Non-scam conversations: {len(labels) - sum(labels)}")

    print("\nTraining TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
    )

    X = vectorizer.fit_transform(texts)

    print(f"TF-IDF matrix: {X.shape}")

    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(X, labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vectorizer_path = os.path.join(
        OUTPUT_DIR,
        "tfidf_vectorizer.joblib"
    )

    classifier_path = os.path.join(
        OUTPUT_DIR,
        "logistic_regression.joblib"
    )

    dump(vectorizer, vectorizer_path)
    dump(classifier, classifier_path)

    print("\nTraining complete.")
    print(f"Saved vectorizer: {vectorizer_path}")
    print(f"Saved classifier: {classifier_path}")


if __name__ == "__main__":
    main()
