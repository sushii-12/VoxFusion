import sys
from pathlib import Path
from .schemas import AASISTResult

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_detection import AASISTDetector

_detector_instance = None


def get_aasist_detector() -> AASISTDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AASISTDetector()
    return _detector_instance


class AASISTAdapter:
    """
    Adapter bridging the FastAPI backend to the authoritative
    src/deepfake_detection.py AASISTDetector.
    """

    def __init__(self):
        self._detector = None

    @property
    def detector(self) -> AASISTDetector:
        if self._detector is None:
            self._detector = get_aasist_detector()
        return self._detector

    def is_available(self) -> bool:
        try:
            return self.detector is not None and self.detector.model is not None
        except Exception:
            return False

    def analyze(self, audio_path: str) -> AASISTResult:
        try:
            res = self.detector.predict(str(audio_path))
            return AASISTResult(
                spoof_score=float(res["deepfake_score"]),
                authentic_score=float(res["bona_fide_score"]),
                raw_bona_fide_score=float(res.get("raw_bona_fide_score", 0.0)),
                prediction=res.get("prediction"),
                available=True,
                detail=f"Prediction: {res.get('prediction')} (deepfake score: {float(res['deepfake_score']):.4f})",
            )
        except Exception as e:
            return AASISTResult(
                available=False,
                detail=f"AASIST inference error: {str(e)}",
            )
