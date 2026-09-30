import torch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

class SpeakerVerifier:
    def __init__(self):
        print("Loading ECAPA-TDNN...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa",
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY
        )

        print(f"ECAPA-TDNN loaded successfully on {self.device}.")

    def verify(self, reference_audio, test_audio):
        score, prediction = self.model.verify_files(
            reference_audio,
            test_audio
        )

        return {
            "voice_match_score": float(score.item()),
            "same_speaker": bool(prediction.item())
        }


if __name__ == "__main__":
    verifier = SpeakerVerifier()

    result = verifier.verify(
        "data/raw/test.wav",
        "data/processed/test_gsm.wav"
    )

    print("\n========== ECAPA RESULT ==========")
    print(f"Voice Match Score : {result['voice_match_score']:.4f}")
    print(f"Same Speaker      : {result['same_speaker']}")
    print("==================================")