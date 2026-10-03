class FusionEngine:
    """
    Combines speaker verification, deepfake detection,
    and scam-intent analysis into a provisional risk score.

    NOTE:
    These weights and thresholds are provisional.
    They must be calibrated using evaluation data.
    """

    def __init__(self):
        self.deepfake_weight = 40
        self.speaker_mismatch_weight = 30
        self.intent_weight = 0.30

    def analyze(
        self,
        speaker_result,
        deepfake_result,
        intent_result
    ):
        risk_score = 0.0
        reasons = []

        # 1. AASIST deepfake detection
        if deepfake_result is not None:
            deepfake_score = float(
                deepfake_result["deepfake_score"]
            )

            deepfake_score = max(
                0.0,
                min(1.0, deepfake_score)
            )

            risk_score += (
                deepfake_score * self.deepfake_weight
            )

            if deepfake_score >= 0.5:
                reasons.append(
                    "AASIST indicates potential synthetic audio."
                )
            else:
                reasons.append(
                    "AASIST indicates audio may be genuine."
                )

        # 2. ECAPA-TDNN speaker verification
        if speaker_result is not None:
            status = speaker_result.get("status")
            same_speaker = speaker_result.get("same_speaker")

            if status == "verified" and same_speaker:
                reasons.append(
                    "Voice matches the registered speaker."
                )

            elif status == "different_speaker" and not same_speaker:
                risk_score += self.speaker_mismatch_weight

                reasons.append(
                    "Voice does not match the registered speaker."
                )

            else:
                reasons.append(
                    "Speaker verification not verified: "
                    "no suitable reference or verification result."
                )
        else:
            reasons.append(
                "Speaker verification unavailable: "
                "no reference recording provided."
            )

        # 3. Scam-intent analysis
        if intent_result is not None:
            intent_score = intent_result.get(
                "scam_intent_score",
                intent_result.get("score")
            )

            if intent_score is None:
                raise KeyError(
                    "Could not find intent score. "
                    f"Available keys: {list(intent_result.keys())}"
                )

            intent_score = float(intent_score)

            # Intent score is expected on a 0-100 scale
            intent_score = max(
                0.0,
                min(100.0, intent_score)
            )

            risk_score += (
                intent_score * self.intent_weight
            )

            # V1 rule-based intent categories
            for category in intent_result.get(
                "matched_categories",
                []
            ):
                reasons.append(
                    f"Scam-related language detected: {category}."
                )

            # V2 ML intent result
            if intent_result.get("status") == "success":
                if "scam_probability" in intent_result:
                    probability = float(
                        intent_result["scam_probability"]
                    )

                    reasons.append(
                        f"V2 scam-intent model probability: "
                        f"{probability:.2f}."
                    )

        else:
            reasons.append(
                "Scam-intent analysis unavailable."
            )

        # Keep final score within 0-100
        risk_score = round(
            min(max(risk_score, 0.0), 100.0),
            2
        )

        # Assign risk level
        if risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": reasons,
            "fusion_status": "provisional"
        }