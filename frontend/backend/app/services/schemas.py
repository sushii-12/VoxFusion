from dataclasses import dataclass, field
from typing import Any


@dataclass
class AASISTResult:
    spoof_score: float | None = None
    authentic_score: float | None = None
    raw_bona_fide_score: float | None = None
    prediction: str | None = None
    available: bool = False
    detail: str = "AASIST adapter not connected."


@dataclass
class ECAPAResult:
    similarity_score: float | None = None
    same_speaker: bool = False
    status: str = "not_verified"
    available: bool = False
    detail: str = "ECAPA adapter not connected."


@dataclass
class WhisperResult:
    transcript: str | None = None
    speech_score: float | None = None
    language: str | None = None
    language_probability: float | None = None
    available: bool = False
    detail: str = "Whisper adapter not connected."


@dataclass
class ScamIntentV1Result:
    scam_intent_score: float = 0.0
    risk_level: str = "LOW"
    matched_categories: list[str] = field(default_factory=list)
    matched_indicators: dict = field(default_factory=dict)
    available: bool = False
    detail: str = ""


@dataclass
class ScamIntentV2Result:
    scam_intent_score: float = 0.0
    scam_probability: float = 0.0
    risk_level: str = "LOW"
    status: str = "success"
    available: bool = False
    detail: str = ""


@dataclass
class FusionResult:
    risk_score: float | None = None
    risk_level: str | None = None
    reasons: list[str] = field(default_factory=list)
    verdict: str | None = None
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    experimental: bool = True
    fusion_status: str = "provisional"


def result_dict(result: Any) -> dict:
    if hasattr(result, "__dict__"):
        return result.__dict__
    return dict(result)
