from pathlib import Path

from faster_whisper import WhisperModel


class WhisperTranscriber:
    def __init__(self):
        print("Loading Whisper...")

        self.device = "cpu"
        self.compute_type = "int8"

        self.model = WhisperModel(
            "small",
            device=self.device,
            compute_type=self.compute_type
        )

        print("Whisper loaded successfully.")

    def transcribe(self, audio_path):
        audio_path = str(Path(audio_path))

        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": float(info.language_probability)
        }


if __name__ == "__main__":
    transcriber = WhisperTranscriber()

    result = transcriber.transcribe(
        "data/processed/test_gsm.wav"
    )

    print("\n========== WHISPER RESULT ==========")
    print(f"Language            : {result['language']}")
    print(f"Language Probability: {result['language_probability']:.4f}")
    print(f"Transcript          : {result['transcript']}")
    print("====================================")
