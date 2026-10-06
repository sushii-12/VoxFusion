class FusionEngine:
    """
    Interaction-based multimodal fusion for VoxFusion.

    AASIST and scam intent are the primary risk drivers.
    ECAPA-TDNN provides speaker-identity context rather than
    independently adding a fixed risk penalty.

    Provisional scoring:
        AASIST: up to 35 points
        Scam intent: up to 30 points
        ECAPA interaction:
            - spoof + enrolled-speaker match: +12
            - spoof + speaker mismatch: +6

    The ECAPA interaction is only applied when both:
        1. AASIST indicates potential spoofing, and
        2. scam intent is meaningfully elevated.

    These weights and thresholds are provisional and are not
    calibrated probabilities.
    """

    def __init__(self):
        # Primary risk drivers
        self.deepfake_weight = 35.0
        self.intent_weight = 0.30

        # ECAPA is a contextual modifier, not an independent vote.
        self.spoof_match_bonus = 12.0
        self.spoof_mismatch_bonus = 6.0

        # Intent level required before the ECAPA × AASIST
        # cloned-voice interaction is activated.
        self.intent_interaction_threshold = 60.0

    def analyze(
        self,
        speaker_result,
        deepfake_result,
        intent_result
    ):
        risk_score = 0.0
        reasons = []

        # ---------------------------------------------------------
        # 1. AASIST deepfake detection
        # ---------------------------------------------------------
        deepfake_score = 0.0
        is_spoof = False

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

            is_spoof = deepfake_score >= 0.5

            if is_spoof:
                reasons.append(
                    "AASIST indicates potential synthetic audio."
                )
            else:
                reasons.append(
                    "AASIST indicates audio may be genuine."
                )

        else:
            reasons.append(
                "AASIST analysis unavailable."
            )

        # ---------------------------------------------------------
        # 2. ECAPA-TDNN speaker verification
        # ---------------------------------------------------------
        speaker_state = "unavailable"

        if speaker_result is not None:
            status = speaker_result.get("status")
            same_speaker = speaker_result.get("same_speaker")

            if status == "verified" and same_speaker:
                speaker_state = "match"

                reasons.append(
                    "Voice matches the registered speaker."
                )

            elif status == "different_speaker" and not same_speaker:
                speaker_state = "mismatch"

                reasons.append(
                    "Voice does not match the registered speaker."
                )

            else:
                reasons.append(
                    "Speaker verification unavailable: "
                    "no suitable reference or verification result."
                )

        else:
            reasons.append(
                "Speaker verification unavailable: "
                "no reference recording provided."
            )

        # ---------------------------------------------------------
        # 3. Scam-intent analysis
        # ---------------------------------------------------------
        intent_score = 0.0

        if intent_result is not None:
            raw_intent_score = intent_result.get(
                "scam_intent_score",
                intent_result.get("score")
            )

            if raw_intent_score is None:
                raise KeyError(
                    "Could not find intent score. "
                    f"Available keys: {list(intent_result.keys())}"
                )

            intent_score = float(raw_intent_score)

            # Intent score is expected on a 0-100 scale.
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

        # ---------------------------------------------------------
        # 4. ECAPA × AASIST × Intent interaction
        #
        # ECAPA does NOT independently increase risk.
        #
        # The interaction is activated only when:
        #   AASIST indicates spoofing
        #   AND scam intent is meaningfully elevated.
        #
        # This prevents an AASIST false positive on a benign GSM
        # conversation from automatically becoming a cloned-voice
        # impersonation alert merely because ECAPA matches.
        # ---------------------------------------------------------
        if (
            is_spoof
            and intent_score >= self.intent_interaction_threshold
        ):
            if speaker_state == "match":
                risk_score += self.spoof_match_bonus

                reasons.append(
                    "Synthetic-audio evidence matches the registered "
                    "speaker while scam-related intent is elevated, "
                    "indicating possible cloned-voice impersonation."
                )

            elif speaker_state == "mismatch":
                risk_score += self.spoof_mismatch_bonus

                reasons.append(
                    "Synthetic-audio evidence comes from a voice that "
                    "does not match the registered speaker while "
                    "scam-related intent is elevated."
                )

        elif is_spoof and speaker_state == "mismatch":
            reasons.append(
                "Speaker mismatch is treated as contextual information; "
                "no additional mismatch penalty is applied without "
                "elevated scam intent."
            )

        elif not is_spoof and speaker_state == "mismatch":
            reasons.append(
                "Speaker mismatch is treated as contextual information "
                "because the audio is not currently flagged as synthetic."
            )

        # ---------------------------------------------------------
        # 5. Final risk score
        # ---------------------------------------------------------
        risk_score = round(
            min(max(risk_score, 0.0), 100.0),
            2
        )

        # Provisional risk thresholds.
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
            "fusion_status": "provisional_interaction_based"
        }