import re


class ScamIntentAnalyzer:
    def __init__(self):
        self.patterns = {
            "emergency": [
                r"\bemergency\b",
                r"\baccident\b",
                r"\bin trouble\b",
                r"\bhelp me\b",
                r"\bhospital\b",
                r"\bin danger\b",
            ],
            "money": [
                r"\bmoney\b",
                r"\btransfer\b",
                r"\bpayment\b",
                r"\bbank account\b",
                r"\bsend cash\b",
                r"\bsend money\b",
            ],
            "urgency": [
                r"\bimmediately\b",
                r"\bright now\b",
                r"\burgent\b",
                r"\bquickly\b",
                r"\bright away\b",
                r"\bdon't wait\b",
            ],
            "secrecy": [
                r"\bdon't tell anyone\b",
                r"\bdo not tell anyone\b",
                r"\bkeep it secret\b",
                r"\bkeep this between us\b",
                r"\btell nobody\b",
                r"\bkeep it private\b",
            ],
        }

        self.weights = {
            "emergency": 25,
            "money": 30,
            "urgency": 25,
            "secrecy": 20,
        }

    def analyze(self, transcript):
        text = transcript.lower()

        matched_indicators = {}
        score = 0

        for category, patterns in self.patterns.items():
            matches = [
                pattern
                for pattern in patterns
                if re.search(pattern, text)
            ]

            if matches:
                matched_indicators[category] = [
                    pattern.replace(r"\b", "").replace("\\", "")
                    for pattern in matches
                ]
                score += self.weights[category]

        score = min(score, 100)

        if score >= 60:
            risk_level = "HIGH"
        elif score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "scam_intent_score": score,
            "risk_level": risk_level,
            "matched_categories": list(matched_indicators.keys()),
            "matched_indicators": matched_indicators,
        }