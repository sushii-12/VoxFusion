import sys
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


# Add AASIST repository to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AASIST_ROOT = PROJECT_ROOT / "aasist"

sys.path.insert(0, str(AASIST_ROOT))

from models.AASIST import Model


# AASIST configuration
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

MODEL_PATH = PROJECT_ROOT / "models" / "AASIST_gsm_finetuned_best.pth"

TARGET_SR = 16000
NUM_SAMPLES = 64600


class AASISTDetector:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading AASIST model...")

        self.model = Model(MODEL_CONFIG).to(self.device)

        state_dict = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("AASIST model loaded successfully.")

    def preprocess_audio(self, audio_path):
        """
        Load audio, convert it to mono, resample to 16 kHz,
        and pad or crop it to AASIST's expected input length.
        """

        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert stereo/multichannel audio to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Convert audio to float32
        audio = audio.astype(np.float32)

        # Reject empty audio
        if audio.size == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        # Resample to 16 kHz without using Librosa or Numba
        if sr != TARGET_SR:
            divisor = gcd(int(sr), TARGET_SR)

            up = TARGET_SR // divisor
            down = int(sr) // divisor

            audio = resample_poly(
                audio,
                up,
                down
            ).astype(np.float32)

        # AASIST expects exactly 64,600 samples
        if len(audio) < NUM_SAMPLES:
            num_repeats = (
                NUM_SAMPLES // len(audio)
            ) + 1

            audio = np.tile(
                audio,
                num_repeats
            )[:NUM_SAMPLES]

        else:
            audio = audio[:NUM_SAMPLES]

        # Convert to a PyTorch tensor
        audio = torch.tensor(
            audio,
            dtype=torch.float32
        ).unsqueeze(0)

        return audio

    def predict(self, audio_path):
        """
        Run AASIST inference.

        Returns:
            bona_fide_score: Softmax probability of class 1.
            deepfake_score: Softmax probability of class 0.
            raw_bona_fide_score: Raw class-1 CM score.
            prediction: bona_fide or deepfake.
        """

        audio = self.preprocess_audio(audio_path)
        audio = audio.to(self.device)

        with torch.no_grad():
            _, output = self.model(audio)

            # Official AASIST raw countermeasure score
            raw_bona_fide_score = output[:, 1].item()

            # Convert logits to probabilities
            probabilities = F.softmax(
                output,
                dim=1
            )

            deepfake_score = probabilities[:, 0].item()
            bona_fide_score = probabilities[:, 1].item()

        prediction = (
            "bona_fide"
            if bona_fide_score >= 0.5
            else "deepfake"
        )

        return {
            "bona_fide_score": bona_fide_score,
            "deepfake_score": deepfake_score,
            "raw_bona_fide_score": raw_bona_fide_score,
            "prediction": prediction,
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.deepfake_detection <audio_path>")
        raise SystemExit(1)

    test_audio = Path(sys.argv[1])

    detector = AASISTDetector()
    result = detector.predict(test_audio)

    print("\n========== AASIST RESULT ==========")
    print(f"Audio: {test_audio}")
    print(
        f"Bona-fide score : "
        f"{result['bona_fide_score']:.4f}"
    )
    print(
        f"Deepfake score  : "
        f"{result['deepfake_score']:.4f}"
    )
    print(
        f"Raw CM score    : "
        f"{result['raw_bona_fide_score']:.4f}"
    )
    print(
        f"Prediction      : "
        f"{result['prediction']}"
    )
    print("===================================")