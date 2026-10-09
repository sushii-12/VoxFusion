from pathlib import Path

from joblib import load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "scam_intent_v2"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"

CLASSIFIER_PATH = MODEL_DIR / "logistic_regression.joblib"


class ScamIntentV2Analyzer:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.load_error = None
        try:
            self.vectorizer = load(VECTORIZER_PATH)
            self.classifier = load(CLASSIFIER_PATH)
        except Exception as e:
            self.load_error = str(e)
            print(f"[ScamIntentV2Analyzer] Loading error: {self.load_error}")

    def analyze(self, transcript):
        # Handle empty transcript early
        if not transcript or not transcript.strip():
            return {
                "scam_intent_score": 0.0,
                "scam_probability": 0.0,
                "risk_level": "LOW",
                "status": "error",
                "message": "Transcript is empty.",
            }
        # If model failed to load, return an error status
        if self.load_error is not None or self.vectorizer is None or self.classifier is None:
            return {
                "scam_intent_score": 0.0,
                "scam_probability": 0.0,
                "risk_level": "LOW",
                "status": "error",
                "message": f"Model load failed: {self.load_error or 'unknown'}",
            }
        # Normal prediction path
        X = self.vectorizer.transform([transcript])
        probability = float(self.classifier.predict_proba(X)[0][1])
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