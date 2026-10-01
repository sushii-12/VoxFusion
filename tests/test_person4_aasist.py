from pathlib import Path
import math

import numpy as np
import pytest
import soundfile as sf

from src.deepfake_detection import AASISTDetector, MODEL_PATH


pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=(
        f"AASIST checkpoint missing: {MODEL_PATH}. "
        "Set up the checkpoint before running AASIST tests."
    ),
)


def create_test_audio(path: Path, sample_rate: int, duration: float, channels: int):
    num_samples = int(sample_rate * duration)
    audio = np.zeros((num_samples, channels), dtype=np.float32)

    t = np.linspace(0, duration, num_samples, endpoint=False)

    if channels == 1:
        audio[:, 0] = 0.1 * np.sin(2 * np.pi * 440 * t)
    else:
        audio[:, 0] = 0.1 * np.sin(2 * np.pi * 440 * t)
        audio[:, 1] = 0.1 * np.sin(2 * np.pi * 550 * t)

    sf.write(path, audio, sample_rate)


def check_aasist_result(result):
    assert set(result.keys()) == {
        "bona_fide_score",
        "deepfake_score",
        "raw_bona_fide_score",
        "prediction",
    }

    assert math.isfinite(result["bona_fide_score"])
    assert math.isfinite(result["deepfake_score"])
    assert math.isfinite(result["raw_bona_fide_score"])

    assert 0.0 <= result["bona_fide_score"] <= 1.0
    assert 0.0 <= result["deepfake_score"] <= 1.0

    assert abs(
        result["bona_fide_score"]
        + result["deepfake_score"]
        - 1.0
    ) < 1e-5

    assert result["prediction"] in {
        "bona_fide",
        "deepfake",
    }


def test_aasist_generated_audio(tmp_path):
    audio = tmp_path / "generated_test.wav"

    create_test_audio(
        audio,
        sample_rate=16000,
        duration=1.0,
        channels=1,
    )

    detector = AASISTDetector()
    result = detector.predict(str(audio))

    check_aasist_result(result)


def test_aasist_generated_phone_style_audio(tmp_path):
    audio = tmp_path / "generated_phone_style_test.wav"

    create_test_audio(
        audio,
        sample_rate=8000,
        duration=1.0,
        channels=2,
    )

    detector = AASISTDetector()
    result = detector.predict(str(audio))

    check_aasist_result(result)
