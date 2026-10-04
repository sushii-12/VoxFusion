"""
VoxFusion ECAPA threshold calibration.

Input CSV format:

reference,test,label
data/eval/spk01_ref.wav,data/eval/spk01_test.wav,1
data/eval/spk01_ref.wav,data/eval/spk02_test.wav,0

label:
    1 = same speaker
    0 = different speaker

The script:
    - computes ECAPA similarity scores
    - calculates FAR
    - calculates FRR
    - estimates EER
    - reports the threshold where FAR and FRR are closest
    - writes all scores to a CSV file

IMPORTANT:
Do threshold selection on a validation set.
Use a separate held-out test set for final reporting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from src.speaker_verification import (
    SpeakerVerifier,
    DEFAULT_VERIFICATION_THRESHOLD,
)


@dataclass
class Trial:
    reference: str
    test: str
    label: int


@dataclass
class TrialResult:
    reference: str
    test: str
    label: int
    score: float


def read_trials(
    csv_path: Path,
) -> List[Trial]:

    trials: List[Trial] = []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "reference",
            "test",
            "label",
        }

        if not required_columns.issubset(
            reader.fieldnames or set()
        ):
            raise ValueError(
                "CSV must contain columns: "
                "reference,test,label"
            )

        for row in reader:

            label = int(
                row["label"].strip()
            )

            if label not in (0, 1):
                raise ValueError(
                    "Labels must be 0 or 1."
                )

            trials.append(
                Trial(
                    reference=row["reference"].strip(),
                    test=row["test"].strip(),
                    label=label,
                )
            )

    if not trials:
        raise ValueError(
            "No trials were found in the CSV."
        )

    return trials


def calculate_rates(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> tuple[float, float]:

    genuine = labels == 1
    impostor = labels == 0

    genuine_count = int(
        genuine.sum()
    )

    impostor_count = int(
        impostor.sum()
    )

    predictions = (
        scores >= threshold
    )

    false_accepts = int(
        np.logical_and(
            predictions,
            impostor,
        ).sum()
    )

    false_rejects = int(
        np.logical_and(
            ~predictions,
            genuine,
        ).sum()
    )

    far = (
        false_accepts / impostor_count
        if impostor_count
        else 0.0
    )

    frr = (
        false_rejects / genuine_count
        if genuine_count
        else 0.0
    )

    return far, frr


def find_eer_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, float]:

    unique_scores = np.unique(
        scores
    )

    # Add thresholds slightly below and above
    # observed scores.
    candidate_thresholds = np.concatenate(
        [
            [unique_scores[0] - 1e-6],
            unique_scores,
            [unique_scores[-1] + 1e-6],
        ]
    )

    best_threshold = None
    best_far = None
    best_frr = None
    best_gap = float("inf")

    for threshold in candidate_thresholds:

        far, frr = calculate_rates(
            scores,
            labels,
            float(threshold),
        )

        gap = abs(
            far - frr
        )

        if gap < best_gap:

            best_gap = gap
            best_threshold = float(
                threshold
            )
            best_far = far
            best_frr = frr

    assert best_threshold is not None
    assert best_far is not None
    assert best_frr is not None

    eer = (
        best_far + best_frr
    ) / 2.0

    return (
        best_threshold,
        eer,
        best_far,
        best_frr,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate ECAPA speaker verification "
            "threshold."
        )
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to labelled pairs CSV.",
    )

    parser.add_argument(
        "--output",
        default="results/ecapa_scores.csv",
        help=(
            "Where to save individual trial scores."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_VERIFICATION_THRESHOLD,
        help=(
            "Current threshold to evaluate "
            "before calibration."
        ),
    )

    args = parser.parse_args()

    pairs_path = Path(
        args.pairs
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials = read_trials(
        pairs_path
    )

    print(
        f"Loaded {len(trials)} speaker trials."
    )

    verifier = SpeakerVerifier(
        threshold=args.threshold
    )

    results: List[TrialResult] = []

    # --------------------------------------------------------
    # Score every trial
    # --------------------------------------------------------

    for index, trial in enumerate(
        trials,
        start=1,
    ):

        print(
            f"[{index}/{len(trials)}] "
            f"{trial.reference} "
            f"vs "
            f"{trial.test}"
        )

        try:

            score = verifier.score(
                trial.reference,
                trial.test,
            )

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )

            continue

        results.append(
            TrialResult(
                reference=trial.reference,
                test=trial.test,
                label=trial.label,
                score=score,
            )
        )

    if not results:
        raise RuntimeError(
            "No trials were successfully scored."
        )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    scores = np.asarray(
        [
            result.score
            for result in results
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            result.label
            for result in results
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Current threshold metrics
    # --------------------------------------------------------

    current_far, current_frr = (
        calculate_rates(
            scores,
            labels,
            args.threshold,
        )
    )

    # --------------------------------------------------------
    # EER threshold
    # --------------------------------------------------------

    (
        eer_threshold,
        eer,
        eer_far,
        eer_frr,
    ) = find_eer_threshold(
        scores,
        labels,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print("ECAPA CALIBRATION RESULTS")
    print(
        "========================================"
    )

    print(
        f"Trials scored       : {len(results)}"
    )

    print(
        f"Current threshold   : "
        f"{args.threshold:.6f}"
    )

    print(
        f"Current FAR         : "
        f"{current_far:.4f}"
    )

    print(
        f"Current FRR         : "
        f"{current_frr:.4f}"
    )

    print(
        f"EER threshold       : "
        f"{eer_threshold:.6f}"
    )

    print(
        f"EER                 : "
        f"{eer:.4f}"
    )

    print(
        f"FAR at EER threshold: "
        f"{eer_far:.4f}"
    )

    print(
        f"FRR at EER threshold: "
        f"{eer_frr:.4f}"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Save score table
    # --------------------------------------------------------

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "reference",
                "test",
                "label",
                "score",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.reference,
                    result.test,
                    result.label,
                    f"{result.score:.8f}",
                ]
            )

    print(
        f"\nScores written to: "
        f"{output_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not automatically use the EER threshold "
        "as your final test-set threshold."
    )

    print(
        "Choose/calibrate the threshold on a validation "
        "set, then evaluate once on a held-out test set."
    )


if __name__ == "__main__":
    main()