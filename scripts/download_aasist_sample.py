from pathlib import Path
import csv
import io

import soundfile as sf
from datasets import load_dataset, Audio


DATASETS = {
    "2019": "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
    "2021": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA",
}

SAMPLES_PER_CLASS = 100
OUTPUT_ROOT = Path("data/aaspoof_sample")


def download_sample(year, dataset_id):
    print(f"\nLoading ASVspoof {year} LA...")

    ds = load_dataset(
        dataset_id,
        split="test",
        streaming=True,
    )

    # Disable automatic decoding, which was triggering librosa/Numba.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Shuffle a limited buffer for a more varied sample.
    ds = ds.shuffle(seed=42, buffer_size=2000)

    audio_dir = OUTPUT_ROOT / year / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    counts = {"bonafide": 0, "spoof": 0}
    rows = []

    for item in ds:
        label_id = item["label"]

        if label_id == 0:
            label = "bonafide"
        elif label_id == 1:
            label = "spoof"
        else:
            continue

        if counts[label] >= SAMPLES_PER_CLASS:
            continue

        audio = item["audio"]
        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")

        # Read the audio without invoking Hugging Face's decoder.
        if audio_bytes is not None:
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
            )
        elif audio_path and Path(audio_path).is_file():
            waveform, sample_rate = sf.read(
                audio_path,
                dtype="float32",
            )
        else:
            print(f"Skipping {item.get('path', 'unknown')}: no audio data")
            continue

        number = counts[label] + 1
        filename = f"{label}_{number:03d}.wav"
        output_path = audio_dir / filename

        sf.write(output_path, waveform, sample_rate)

        rows.append({
            "filename": filename,
            "label": label,
            "sample_rate": sample_rate,
            "source": item.get("path") or audio_path or "",
        })

        counts[label] += 1

        print(
            f"{year}: {label} "
            f"{counts[label]}/{SAMPLES_PER_CLASS}"
        )

        if all(
            count == SAMPLES_PER_CLASS
            for count in counts.values()
        ):
            break

    labels_path = OUTPUT_ROOT / year / "labels.csv"

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label",
                "sample_rate",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nASVspoof {year} result:")
    print(f"  Bona fide: {counts['bonafide']}")
    print(f"  Spoof:     {counts['spoof']}")
    print(f"  Labels:    {labels_path}")

    if any(
        count != SAMPLES_PER_CLASS
        for count in counts.values()
    ):
        raise RuntimeError(
            f"Could not collect enough samples for {year}. "
            "Check the output and dataset access."
        )


if __name__ == "__main__":
    for year, dataset_id in DATASETS.items():
        download_sample(year, dataset_id)

    print("\nBoth dataset samples downloaded successfully.")