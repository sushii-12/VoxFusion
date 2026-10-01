import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scam_intent import ScamIntentAnalyzer


def check(name, transcript, expected_score, expected_risk, excluded_categories=()):
    analyzer = ScamIntentAnalyzer()
    result = analyzer.analyze(transcript)

    assert result["scam_intent_score"] == expected_score, (
        f"{name}: expected score {expected_score}, "
        f"got {result['scam_intent_score']}"
    )

    assert result["risk_level"] == expected_risk, (
        f"{name}: expected risk {expected_risk}, "
        f"got {result['risk_level']}"
    )

    for category in excluded_categories:
        assert category not in result["matched_categories"], (
            f"{name}: unexpected category {category}"
        )

    print(f"PASS: {name}")


check(
    "Clear scam-style call",
    "There is an emergency. Send money immediately and do not tell anyone.",
    100,
    "HIGH",
)

check(
    "Negated emergency",
    "There is no emergency. Send money immediately.",
    55,
    "MEDIUM",
    excluded_categories=("emergency",),
)

check(
    "Normal family conversation",
    "Hi, I am on my way home. I will call you tonight.",
    0,
    "LOW",
)

check(
    "Benign secrecy context",
    "Please keep this between us because it is a birthday surprise.",
    0,
    "LOW",
)

print("All Person 4 scam-intent regression tests passed.")
