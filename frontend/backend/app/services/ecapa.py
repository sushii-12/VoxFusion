import sys
from pathlib import Path
from .schemas import ECAPAResult

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speaker_verification import SpeakerVerifier

_verifier_instance = None


def get_speaker_verifier() -> SpeakerVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = SpeakerVerifier()
    return _verifier_instance


class ECAPAAdapter:
    """
    Adapter bridging the FastAPI backend to the authoritative
    src/speaker_verification.py SpeakerVerifier.
    """

    def __init__(self):
        self._verifier = None

    @property
    def verifier(self) -> SpeakerVerifier:
        if self._verifier is None:
            self._verifier = get_speaker_verifier()
        return self._verifier

    def is_available(self) -> bool:
        try:
            return self.verifier is not None and self.verifier.model is not None
        except Exception:
            return False

    def compare(self, audio_path: str, reference_paths: list[str] | None = None) -> ECAPAResult:
        if not reference_paths:
            return ECAPAResult(
                similarity_score=None,
                same_speaker=False,
                status="not_verified",
                available=True,
                detail="No reference recording provided for speaker verification.",
            )

        ref_path = reference_paths[0]
        try:
            res = self.verifier.verify(str(ref_path), str(audio_path))
            status = res.get("status", "error")
            raw_score = res.get("voice_match_score")
            score = float(raw_score) if raw_score is not None else None
            same_speaker = bool(res.get("same_speaker", False))
            message = res.get("message", "")

            return ECAPAResult(
                similarity_score=score,
                same_speaker=same_speaker,
                status=status,
                available=(status != "error"),
                detail=message or f"Status: {status}",
            )
        except Exception as e:
            return ECAPAResult(
                similarity_score=None,
                same_speaker=False,
                status="error",
                available=False,
                detail=f"ECAPA verification error: {str(e)}",
            )
