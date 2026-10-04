import sys
from pathlib import Path
from .schemas import WhisperResult

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transcription import WhisperTranscriber

_transcriber_instance = None


def get_whisper_transcriber() -> WhisperTranscriber:
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = WhisperTranscriber()
    return _transcriber_instance


class WhisperAdapter:
    """
    Adapter bridging the FastAPI backend to the authoritative
    src/transcription.py WhisperTranscriber.
    """

    def __init__(self):
        self._transcriber = None

    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = get_whisper_transcriber()
        return self._transcriber

    def is_available(self) -> bool:
        try:
            return self.transcriber is not None and self.transcriber.model is not None
        except Exception:
            return False

    def transcribe(self, audio_path: str) -> WhisperResult:
        try:
            res = self.transcriber.transcribe(str(audio_path))
            if res.get("status") == "error":
                return WhisperResult(
                    transcript="",
                    speech_score=0.0,
                    language=None,
                    language_probability=0.0,
                    available=False,
                    detail=res.get("error", "Transcription failed."),
                )

            transcript = res.get("transcript", "")
            lang = res.get("language", "en")
            prob = float(res.get("language_probability", 1.0))

            return WhisperResult(
                transcript=transcript,
                speech_score=prob,
                language=lang,
                language_probability=prob,
                available=True,
                detail=f"Transcribed ({lang})" if transcript else "No speech detected",
            )
        except Exception as e:
            return WhisperResult(
                transcript="",
                speech_score=0.0,
                available=False,
                detail=f"Whisper transcription error: {str(e)}",
            )
