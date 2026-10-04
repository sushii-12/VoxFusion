from pathlib import Path

import pytest

from src.speaker_verification import SpeakerVerifier


DATA_DIR = Path("data/raw")

REFERENCE = DATA_DIR / "reference.wav"
SAME_SPEAKER = DATA_DIR / "same_speaker.wav"
DIFFERENT_SPEAKER = DATA_DIR / "different_speaker.wav"


@pytest.fixture(scope="module")
def verifier():
    return SpeakerVerifier(
        threshold=0.25
    )


@pytest.fixture(scope="module")
def audio_files():
    files = [
        REFERENCE,
        SAME_SPEAKER,
        DIFFERENT_SPEAKER,
    ]

    missing = [
        str(path)
        for path in files
        if not path.exists()
    ]

    if missing:
        pytest.skip(
            "Test audio files missing: "
            + ", ".join(missing)
        )

    return files


def test_reference_embedding(
    verifier,
    audio_files,
):

    embedding = (
        verifier.extract_reference_embedding(
            str(REFERENCE)
        )
    )

    assert embedding is not None
    assert embedding.numel() > 0


def test_reference_embedding_is_cached(
    verifier,
    audio_files,
):

    first = (
        verifier.extract_reference_embedding(
            str(REFERENCE)
        )
    )

    second = (
        verifier.extract_reference_embedding(
            str(REFERENCE)
        )
    )

    assert first is second


def test_same_speaker_score(
    verifier,
    audio_files,
):

    result = verifier.verify(
        str(REFERENCE),
        str(SAME_SPEAKER),
    )

    assert result["status"] == "success"

    assert (
        result["voice_match_score"]
        is not None
    )

    assert (
        result["same_speaker"]
        is True
    )


def test_different_speaker_score(
    verifier,
    audio_files,
):

    result = verifier.verify(
        str(REFERENCE),
        str(DIFFERENT_SPEAKER),
    )

    assert result["status"] == "success"

    assert (
        result["voice_match_score"]
        is not None
    )

    assert (
        result["same_speaker"]
        is False
    )


def test_missing_reference(
    verifier,
):

    result = verifier.verify(
        "data/raw/does_not_exist.wav",
        str(SAME_SPEAKER),
    )

    assert result["status"] == "error"

    assert (
        result["voice_match_score"]
        is None
    )


def test_short_reference(
    verifier,
    tmp_path,
):

    import numpy as np
    import soundfile as sf

    short_audio = (
        tmp_path / "short.wav"
    )

    sf.write(
        short_audio,
        np.zeros(
            4000,
            dtype=np.float32,
        ),
        16000,
    )

    result = verifier.verify(
        str(short_audio),
        str(SAME_SPEAKER),
    )

    assert result["status"] == "error"

    assert (
        result["voice_match_score"]
        is None
    )


def test_multiple_reference_recordings(
    verifier,
    audio_files,
):

    embedding = (
        verifier.extract_reference_embedding(
            [
                str(REFERENCE),
                str(SAME_SPEAKER),
            ]
        )
    )

    assert embedding is not None
    assert embedding.numel() > 0


def test_embedding_reuse(
    verifier,
    audio_files,
):

    reference_embedding = (
        verifier.extract_reference_embedding(
            str(REFERENCE)
        )
    )

    result = (
        verifier.verify_with_embedding(
            reference_embedding,
            str(SAME_SPEAKER),
        )
    )

    assert result["status"] == "success"


def test_score_only_api(
    verifier,
    audio_files,
):

    score = verifier.score(
        str(REFERENCE),
        str(SAME_SPEAKER),
    )

    assert isinstance(
        score,
        float,
    )

def test_same_speaker_score_is_higher(
    verifier,
    audio_files,
):
    same_score = verifier.score(
        str(REFERENCE),
        str(SAME_SPEAKER),
    )

    different_score = verifier.score(
        str(REFERENCE),
        str(DIFFERENT_SPEAKER),
    )

    print(
        f"\nSame-speaker score: {same_score:.6f}"
    )

    print(
        f"Different-speaker score: "
        f"{different_score:.6f}"
    )

    assert same_score > different_score