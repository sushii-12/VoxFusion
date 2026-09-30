import re


def is_negated(text, match_start):
    """Check whether an indicator is negated in its local context."""

    preceding_text = text[:match_start]

    # Reset context at sentence boundaries and contrast words.
    context_parts = re.split(
        r"[.!?;]|\bbut\b|\bhowever\b|\balthough\b|\byet\b|\band\b",
        preceding_text
    )

    local_context = context_parts[-1]

    preceding_words = re.findall(
        r"\b[\w']+\b",
        local_context
    )[-5:]

    negation_words = {
        # Basic negation
        "no", "not", "never", "neither", "nor", "none",

        # Common contractions
        "don't", "doesn't", "didn't", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't",
        "can't", "cannot", "couldn't", "won't", "wouldn't",
        "shouldn't",

        # Other negative expressions
        "without", "nobody", "nothing",
    }

    return any(
        word in negation_words
        for word in preceding_words
    )


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
                r"\burgently\b",
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

        # These phrases are positive indicators despite containing
        # negation words.
        self.negation_exempt_patterns = {
            r"\bdon't wait\b",
            r"\bdon't tell anyone\b",
            r"\bdo not tell anyone\b",
        }

        # Selected benign or resolved emergency contexts.
        self.benign_emergency_phrases = [
            "to visit my friend",
            "to visit a friend",
            "everyone is safe now",
            "everyone was safe",
            "no one was hurt",
            "nobody was hurt",
            "last year and recovered",
            "visited the hospital yesterday",
        ]

        # Selected benign secrecy contexts.
        self.benign_secrecy_phrases = [
            "surprise party",
            "birthday surprise",
        ]

    def is_benign_emergency_context(self, text, match):
        """
        Check whether an emergency indicator occurs in a selected
        benign or resolved context.
        """

        # Find the sentence containing the indicator.
        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
        ) + 1

        sentence_end_candidates = [
            pos
            for pos in (
                text.find(".", match.end()),
                text.find("!", match.end()),
                text.find("?", match.end()),
            )
            if pos != -1
        ]

        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates
            else len(text)
        )

        sentence = text[sentence_start:sentence_end].lower()

        return any(
            phrase in sentence
            for phrase in self.benign_emergency_phrases
        )

    def is_benign_secrecy_context(self, text, match):
        """Check whether secrecy occurs in a selected benign context."""

        # Find the sentence containing the secrecy indicator.
        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
        ) + 1

        sentence_end_candidates = [
            pos
            for pos in (
                text.find(".", match.end()),
                text.find("!", match.end()),
                text.find("?", match.end()),
            )
            if pos != -1
        ]

        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates
            else len(text)
        )

        sentence = text[sentence_start:sentence_end].lower()

        return any(
            phrase in sentence
            for phrase in self.benign_secrecy_phrases
        )

    def analyze(self, transcript):
        """Analyze a transcript and return its scam-intent indicators."""

        text = transcript.lower()
        matched_indicators = {}
        score = 0

        for category, patterns in self.patterns.items():
            matches = []

            for pattern in patterns:
                for match in re.finditer(pattern, text):

                    is_exempt = (
                        pattern in self.negation_exempt_patterns
                    )

                    is_valid_indicator = (
                        (
                            is_exempt
                            or not is_negated(text, match.start())
                        )
                        and not (
                            category == "emergency"
                            and self.is_benign_emergency_context(
                                text, match
                            )
                        )
                        and not (
                            category == "secrecy"
                            and self.is_benign_secrecy_context(
                                text, match
                            )
                        )
                    )

                    if is_valid_indicator:
                        matches.append(pattern)
                        break

            if matches:
                matched_indicators[category] = [
                    pattern.replace(r"\b", "").replace("\\", "")
                    for pattern in matches
                ]

                score += self.weights[category]

        # Keep the score within 0–100.
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