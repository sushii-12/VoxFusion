import argparse
import os
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


TARGET_SR = 16000
MIN_REFERENCE_SECONDS = 1.0
VERIFICATION_THRESHOLD = 0.25


class SpeakerVerifier:
    def __init__(self):
        print("Loading ECAPA-TDNN...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa",
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

        self._reference_embedding = None
        self._reference_cache_key = None

        print(f"ECAPA-TDNN loaded successfully on {self.device}.")

    def _load_audio(self, audio_path):
        """Load audio, convert to mono, and resample to 16 kHz."""
        if not audio_path or not os.path.isfile(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as exc:
            raise ValueError(
                f"Could not read audio file: {audio_path}"
            ) from exc

        if audio.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        if sample_rate != TARGET_SR:
            divisor = gcd(int(sample_rate), TARGET_SR)
            up = TARGET_SR // divisor
            down = int(sample_rate) // divisor
            audio = resample_poly(audio, up, down).astype(np.float32)

        if audio.size == 0:
            raise ValueError(
                f"Audio file contains no usable samples: {audio_path}"
            )

        return torch.from_numpy(audio)

    def _encode_audio(self, audio):
        """Extract one ECAPA embedding from a 16 kHz mono waveform."""
        waveform = audio.unsqueeze(0).to(self.device)
        embedding = self.model.encode_batch(waveform)
        return embedding.detach()

    def _reference_key(self, reference_audio):
        """Return a cache key that changes when the reference file changes."""
        try:
            stat = os.stat(reference_audio)
        except OSError:
            return None

        return (
            os.path.abspath(reference_audio),
            stat.st_size,
            stat.st_mtime_ns,
        )

    def extract_reference_embedding(self, reference_audio):
        """
        Extract and cache the reference speaker embedding.

        The cached embedding is reused while the reference file remains
        unchanged during the lifetime of this SpeakerVerifier instance.
        """
        cache_key = self._reference_key(reference_audio)

        if (
            self._reference_embedding is not None
            and cache_key is not None
            and self._reference_cache_key == cache_key
        ):
            return self._reference_embedding

        audio = self._load_audio(reference_audio)

        duration = len(audio) / TARGET_SR
        if duration < MIN_REFERENCE_SECONDS:
            raise ValueError(
                "Reference audio is too short. "
                f"At least {MIN_REFERENCE_SECONDS:.1f} second is required."
            )

        self._reference_embedding = self._encode_audio(audio)
        self._reference_cache_key = self._reference_key(reference_audio)

        return self._reference_embedding

    def verify_with_embedding(self, reference_embedding, test_audio):
        """
        Compare a test recording against a previously extracted reference
        embedding.

        This method allows the application layer to store the reference
        embedding for the active session and reuse it across multiple calls.
        """
        try:
            if reference_embedding is None:
                raise ValueError("Reference speaker embedding is missing.")

            if not isinstance(reference_embedding, torch.Tensor):
                raise ValueError("Reference speaker embedding is invalid.")

            test_audio_tensor = self._load_audio(test_audio)
            test_embedding = self._encode_audio(test_audio_tensor)

            score = self.model.similarity(
                reference_embedding.to(self.device),
                test_embedding,
            )

            score_value = float(score.squeeze().item())
            same_speaker = bool(score_value > VERIFICATION_THRESHOLD)

            return {
                "voice_match_score": score_value,
                "same_speaker": same_speaker,
                "status": (
                    "verified" if same_speaker else "different_speaker"
                ),
                "message": (
                    "Reference and test recordings were compared successfully."
                ),
            }

        except ValueError as exc:
            return {
                "voice_match_score": None,
                "same_speaker": False,
                "status": "error",
                "message": str(exc),
            }

    def verify(self, reference_audio, test_audio):
        """
        Compare a test recording against a reference audio file.

        Preserves the existing verify(reference_audio, test_audio) interface
        used by the application while reusing the cached reference embedding.
        """
        try:
            reference_embedding = self.extract_reference_embedding(
                reference_audio
            )
            return self.verify_with_embedding(
                reference_embedding,
                test_audio,
            )
        except ValueError as exc:
            return {
                "voice_match_score": None,
                "same_speaker": False,
                "status": "error",
                "message": str(exc),
            }


def main():
    parser = argparse.ArgumentParser(
        description="Compare a test recording against a reference speaker."
    )
    parser.add_argument(
        "reference_audio",
        help="Path to the clean reference recording.",
    )
    parser.add_argument(
        "test_audio",
        help="Path to the suspicious/test recording.",
    )
    args = parser.parse_args()

    verifier = SpeakerVerifier()
    result = verifier.verify(
        args.reference_audio,
        args.test_audio,
    )

    print("\n========== ECAPA RESULT ==========")
    print(f"Voice Match Score : {result['voice_match_score']}")
    print(f"Same Speaker      : {result['same_speaker']}")
    print(f"Status            : {result['status']}")
    print(f"Message           : {result['message']}")
    print("==================================")


if __name__ == "__main__":
    main()
