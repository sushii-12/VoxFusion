import sys
from pathlib import Path
from .schemas import (
    AASISTResult,
    ECAPAResult,
    WhisperResult,
    ScamIntentV1Result,
    ScamIntentV2Result,
    FusionResult,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scam_intent import ScamIntentAnalyzer
from src.scam_intent_v2 import ScamIntentV2Analyzer
from src.fusion import FusionEngine

_v1_instance = None
_v2_instance = None
_fusion_instance = None


def get_scam_v1_analyzer() -> ScamIntentAnalyzer:
    global _v1_instance
    if _v1_instance is None:
        _v1_instance = ScamIntentAnalyzer()
    return _v1_instance


def get_scam_v2_analyzer() -> ScamIntentV2Analyzer:
    global _v2_instance
    if _v2_instance is None:
        _v2_instance = ScamIntentV2Analyzer()
    return _v2_instance


def get_fusion_engine() -> FusionEngine:
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = FusionEngine()
    return _fusion_instance


class ScamIntentAdapter:
    def __init__(self):
        self._v1 = None
        self._v2 = None

    @property
    def v1(self) -> ScamIntentAnalyzer:
        if self._v1 is None:
            self._v1 = get_scam_v1_analyzer()
        return self._v1

    @property
    def v2(self) -> ScamIntentV2Analyzer:
        if self._v2 is None:
            self._v2 = get_scam_v2_analyzer()
        return self._v2

    def analyze_v1(self, transcript: str) -> ScamIntentV1Result:
        try:
            res = self.v1.analyze(transcript or "")
            return ScamIntentV1Result(
                scam_intent_score=float(res.get("scam_intent_score", 0.0)),
                risk_level=res.get("risk_level", "LOW"),
                matched_categories=res.get("matched_categories", []),
                matched_indicators=res.get("matched_indicators", {}),
                available=True,
                detail=f"V1 Risk: {res.get('risk_level', 'LOW')} (Score: {res.get('scam_intent_score', 0)})",
            )
        except Exception as e:
            return ScamIntentV1Result(
                available=False,
                detail=f"V1 Intent analysis error: {str(e)}",
            )

    def analyze_v2(self, transcript: str) -> ScamIntentV2Result:
        try:
            res = self.v2.analyze(transcript or "")
            return ScamIntentV2Result(
                scam_intent_score=float(res.get("scam_intent_score", 0.0)),
                scam_probability=float(res.get("scam_probability", 0.0)),
                risk_level=res.get("risk_level", "LOW"),
                status=res.get("status", "success"),
                available=True,
                detail=f"V2 Risk: {res.get('risk_level', 'LOW')} (Score: {res.get('scam_intent_score', 0):.1f})",
            )
        except Exception as e:
            return ScamIntentV2Result(
                available=False,
                detail=f"V2 Intent analysis error: {str(e)}",
            )


class FusionAdapter:
    """
    Adapter bridging the FastAPI backend to the authoritative
    src/fusion.py FusionEngine.
    """

    def __init__(self):
        self._engine = None

    @property
    def engine(self) -> FusionEngine:
        if self._engine is None:
            self._engine = get_fusion_engine()
        return self._engine

    def run_fusion(
        self,
        aasist_result: AASISTResult,
        ecapa_result: ECAPAResult,
        v1_result: ScamIntentV1Result,
        v2_result: ScamIntentV2Result,
    ) -> FusionResult:
        # Build speaker_result payload for FusionEngine
        speaker_dict = None
        if ecapa_result.available and ecapa_result.status != "not_verified":
            speaker_dict = {
                "status": ecapa_result.status,
                "same_speaker": ecapa_result.same_speaker,
                "voice_match_score": ecapa_result.similarity_score,
                "message": ecapa_result.detail,
            }
        elif ecapa_result.status == "not_verified":
            speaker_dict = None  # FusionEngine treats None as "No reference recording provided"

        # Build deepfake_result payload for FusionEngine
        deepfake_dict = None
        if aasist_result.available and aasist_result.spoof_score is not None:
            deepfake_dict = {
                "deepfake_score": aasist_result.spoof_score,
                "bona_fide_score": aasist_result.authentic_score,
                "raw_bona_fide_score": aasist_result.raw_bona_fide_score,
                "prediction": aasist_result.prediction,
            }

        # Build combined intent payload carrying both V1 indicators and V2 probability
        intent_dict = {
            "scam_intent_score": v2_result.scam_intent_score if v2_result.available else v1_result.scam_intent_score,
            "risk_level": v2_result.risk_level if v2_result.available else v1_result.risk_level,
            "matched_categories": v1_result.matched_categories,
            "matched_indicators": v1_result.matched_indicators,
            "scam_probability": v2_result.scam_probability,
            "status": v2_result.status if v2_result.available else "success",
        }

        fused = self.engine.analyze(
            speaker_result=speaker_dict,
            deepfake_result=deepfake_dict,
            intent_result=intent_dict,
        )

        risk_score = float(fused.get("risk_score", 0.0))
        risk_level = fused.get("risk_level", "LOW")
        reasons = fused.get("reasons", [])

        # Construct evidence bullet points
        evidence = []
        if aasist_result.available and aasist_result.spoof_score is not None:
            evidence.append(f"AASIST deepfake score: {aasist_result.spoof_score:.4f} ({aasist_result.prediction})")
        if ecapa_result.available and ecapa_result.similarity_score is not None:
            evidence.append(f"ECAPA similarity score: {ecapa_result.similarity_score:.4f} (Status: {ecapa_result.status})")
        elif ecapa_result.status == "not_verified":
            evidence.append("ECAPA: No reference voice recording provided")
        if v1_result.matched_categories:
            evidence.append(f"V1 Intent categories detected: {', '.join(v1_result.matched_categories)}")
        if v2_result.available:
            evidence.append(f"V2 ML scam probability: {v2_result.scam_probability:.4f} (Risk: {v2_result.risk_level})")

        return FusionResult(
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
            verdict=f"Risk Level: {risk_level} (Score: {risk_score:.1f}/100)",
            confidence=risk_score / 100.0,
            evidence=evidence,
            experimental=True,
            fusion_status=fused.get("fusion_status", "provisional"),
        )


def fuse(aasist: AASISTResult, ecapa: ECAPAResult, whisper: WhisperResult) -> FusionResult:
    """
    Standard fusion entrypoint using authoritative src/fusion.py FusionEngine
    and both V1 & V2 scam intent analyzers.
    """
    scam_adapter = ScamIntentAdapter()
    v1_res = scam_adapter.analyze_v1(whisper.transcript or "")
    v2_res = scam_adapter.analyze_v2(whisper.transcript or "")
    fusion_adapter = FusionAdapter()
    return fusion_adapter.run_fusion(aasist, ecapa, v1_res, v2_res)
