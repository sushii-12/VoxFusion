import os

from joblib import load


MODEL_DIR = os.path.join("models", "scam_intent_v2")

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression.joblib"
)


class ScamIntentV2Analyzer:
    def __init__(self):
        self.vectorizer = load(VECTORIZER_PATH)
        self.classifier = load(CLASSIFIER_PATH)

    def analyze(self, transcript):
        if not transcript or not transcript.strip():
            return {
                "scam_intent_score": 0.0,
                "scam_probability": 0.0,
                "risk_level": "LOW",
                "status": "error",
                "message": "Transcript is empty."
            }

        X = self.vectorizer.transform([transcript])

        probability = float(
            self.classifier.predict_proba(X)[0][1]
        )

        score = probability * 100.0

        if score >= 60:
            risk_level = "HIGH"
        elif score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "scam_intent_score": score,
            "scam_probability": probability,
            "risk_level": risk_level,
            "status": "success",
        }