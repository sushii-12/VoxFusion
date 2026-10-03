import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speaker_verification import SpeakerVerifier


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests" / "fixtures"
REFERENCE_AUDIO = TEST_DIR / "person4_test.wav"
TEST_AUDIO = TEST_DIR / "person4_test_phone.wav"


def create_test_audio():
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    duration = 3

    t = np.linspace(
        0,
        duration,
        sample_rate * duration,
        endpoint=False,
    )

    # Synthetic audio fixture for repeatable testing.
    audio = (
        0.2 * np.sin(2 * np.pi * 180 * t)
        + 0.1 * np.sin(2 * np.pi * 260 * t)
    ).astype(np.float32)

    sf.write(
        REFERENCE_AUDIO,
        audio,
        sample_rate,
    )

    sf.write(
        TEST_AUDIO,
        audio,
        sample_rate,
    )


def test_same_speaker_and_cache():
    create_test_audio()

    verifier = SpeakerVerifier()

    embedding = verifier.extract_reference_embedding(
        str(REFERENCE_AUDIO)
    )

    assert embedding is not None
    assert embedding.shape[-1] == 192

    first = verifier.verify_with_embedding(
        embedding,
        str(TEST_AUDIO),
    )

    assert first["status"] == "verified"
    assert first["same_speaker"] is True
    assert first["voice_match_score"] is not None

    second = verifier.verify(
        str(REFERENCE_AUDIO),
        str(TEST_AUDIO),
    )

    assert second["status"] == "verified"
    assert second["same_speaker"] is True
    assert second["voice_match_score"] == first["voice_match_score"]

    cached = verifier.extract_reference_embedding(
        str(REFERENCE_AUDIO)
    )

    assert cached is embedding

    print(
        "PASS: same-speaker comparison and "
        "reference embedding cache"
    )


def test_error_cases():
    create_test_audio()

    verifier = SpeakerVerifier()

    missing = verifier.verify(
        str(ROOT / "data/raw/does_not_exist.wav"),
        str(TEST_AUDIO),
    )

    assert missing["status"] == "error"
    assert missing["voice_match_score"] is None

    short_path = TEST_DIR / "voxfusion_person3_short.wav"
    invalid_path = TEST_DIR / "voxfusion_person3_invalid.wav"

    sf.write(
        short_path,
        np.zeros(8000, dtype=np.float32),
        16000,
    )

    invalid_path.write_text("not audio")

    short = verifier.verify(
        str(short_path),
        str(TEST_AUDIO),
    )

    assert short["status"] == "error"
    assert "too short" in short["message"]

    invalid = verifier.verify(
        str(invalid_path),
        str(TEST_AUDIO),
    )

    assert invalid["status"] == "error"
    assert "Could not read audio file" in invalid["message"]

    short_path.unlink(missing_ok=True)
    invalid_path.unlink(missing_ok=True)

    print(
        "PASS: missing, short, and invalid reference handling"
    )