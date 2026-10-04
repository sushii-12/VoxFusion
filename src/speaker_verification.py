"""
VoxFusion - ECAPA-TDNN Speaker Verification

Person 3 module:
- Reference audio validation
- ECAPA embedding extraction
- Reference embedding caching
- Single/multiple reference enrollment
- Speaker similarity scoring
- Threshold-based verification
- Reusable reference embeddings
- Robust error handling

The default pretrained model is:
    speechbrain/spkrec-ecapa-voxceleb

Important:
    VERIFICATION_THRESHOLD is a configurable starting value.
    It should be calibrated on a labelled validation set before
    being treated as the final project threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# Configuration
# ============================================================

TARGET_SR = 16_000

MIN_REFERENCE_SECONDS = 1.0
MIN_TEST_SECONDS = 0.5

# Starting threshold only.
# Calibrate this using a labelled validation set before
# using it as your final project threshold.
DEFAULT_VERIFICATION_THRESHOLD = 0.41

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_SAVE_DIR = "models/ecapa"

AudioInput = Union[str, os.PathLike]
ReferenceInput = Union[
    AudioInput,
    Sequence[AudioInput],
]


class SpeakerVerifier:
    """
    ECAPA-TDNN speaker verification engine.

    Supports:
        1. One reference audio file
        2. Multiple reference audio files
        3. Reference embedding caching
        4. Direct embedding reuse
        5. Score-only evaluation
        6. Threshold-based verification
    """

    def __init__(
        self,
        threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
        model_source: str = MODEL_SOURCE,
        model_save_dir: str = MODEL_SAVE_DIR,
    ) -> None:

        if not np.isfinite(threshold):
            raise ValueError("Threshold must be a finite number.")

        self.threshold = float(threshold)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_source = model_source
        self.model_save_dir = model_save_dir

        print("Loading ECAPA-TDNN...")
        print(f"Device: {self.device}")
        print(f"Verification threshold: {self.threshold:.4f}")

        self.model = SpeakerRecognition.from_hparams(
            source=self.model_source,
            savedir=self.model_save_dir,
            run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY,
        )

        # Cached enrollment/profile embedding.
        self._reference_embedding: Optional[torch.Tensor] = None

        # Cache key based on reference file metadata.
        self._reference_cache_key: Optional[
            Tuple[Tuple[str, int, int], ...]
        ] = None

        print(
            f"ECAPA-TDNN loaded successfully on {self.device}."
        )

    # ========================================================
    # Utility / configuration
    # ========================================================

    def set_threshold(self, threshold: float) -> None:
        """
        Update the verification threshold at runtime.
        """

        if not np.isfinite(threshold):
            raise ValueError(
                "Threshold must be a finite number."
            )

        self.threshold = float(threshold)

    def clear_reference_cache(self) -> None:
        """
        Clear the currently cached reference embedding.
        """

        self._reference_embedding = None
        self._reference_cache_key = None

    @staticmethod
    def _as_paths(
        audio_input: ReferenceInput,
    ) -> List[str]:
        """
        Convert a single path or sequence of paths into a list.
        """

        if isinstance(audio_input, (str, os.PathLike)):
            return [os.fspath(audio_input)]

        paths = [
            os.fspath(path)
            for path in audio_input
        ]

        if not paths:
            raise ValueError(
                "At least one reference audio file is required."
            )

        return paths

    # ========================================================
    # Audio loading / preprocessing
    # ========================================================

    def _load_audio(
        self,
        audio_path: AudioInput,
    ) -> torch.Tensor:
        """
        Load an audio file, convert it to mono and resample it
        to 16 kHz.

        Returns:
            torch.Tensor with shape [time]
        """

        audio_path = os.fspath(audio_path)

        if not audio_path:
            raise ValueError(
                "Audio path was not provided."
            )

        if not os.path.isfile(audio_path):
            raise ValueError(
                f"Audio file not found: {audio_path}"
            )

        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as exc:
            raise ValueError(
                f"Could not read audio file: {audio_path}"
            ) from exc

        if audio.size == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        # Stereo / multi-channel -> mono
        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1,
            )

        audio = audio.astype(
            np.float32,
            copy=False,
        )

        # Reject NaN / Inf.
        if not np.isfinite(audio).all():
            raise ValueError(
                f"Audio contains NaN or Inf values: {audio_path}"
            )

        # Resample to 16 kHz when necessary.
        sample_rate = int(sample_rate)

        if sample_rate <= 0:
            raise ValueError(
                f"Invalid sample rate: {sample_rate}"
            )

        if sample_rate != TARGET_SR:

            divisor = gcd(
                sample_rate,
                TARGET_SR,
            )

            up = TARGET_SR // divisor
            down = sample_rate // divisor

            audio = resample_poly(
                audio,
                up,
                down,
            ).astype(
                np.float32,
                copy=False,
            )

        if audio.size == 0:
            raise ValueError(
                f"Audio contains no usable samples: {audio_path}"
            )

        # Prevent malformed values caused by conversion.
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return torch.from_numpy(audio)

    # ========================================================
    # ECAPA embedding
    # ========================================================

    def _encode_audio(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract a single ECAPA speaker embedding.

        The SpeechBrain model expects a batch dimension.
        """

        if not isinstance(audio, torch.Tensor):
            raise ValueError(
                "Audio must be a torch.Tensor."
            )

        if audio.numel() == 0:
            raise ValueError(
                "Audio tensor is empty."
            )

        waveform = (
            audio
            .unsqueeze(0)
            .to(self.device)
        )

        # Explicit inference context:
        # speaker verification does not need gradients.
        with torch.inference_mode():
            embedding = self.model.encode_batch(
                waveform
            )

        embedding = embedding.detach()

        if embedding.numel() == 0:
            raise ValueError(
                "ECAPA produced an empty embedding."
            )

        # Ensure a stable [1, embedding_dim] shape.
        embedding = embedding.reshape(
            embedding.shape[0],
            -1,
        )

        # L2-normalize the embedding.
        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

        return embedding

    # ========================================================
    # Reference cache
    # ========================================================

    @staticmethod
    def _file_cache_key(
        path: AudioInput,
    ) -> Tuple[str, int, int]:
        """
        Build a cache key from:
            absolute path
            file size
            modification time
        """

        path = os.fspath(path)

        try:
            stat = os.stat(path)
        except OSError:
            return (
                os.path.abspath(path),
                -1,
                -1,
            )

        return (
            os.path.abspath(path),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    def _reference_key(
        self,
        reference_audio: ReferenceInput,
    ) -> Tuple[Tuple[str, int, int], ...]:
        """
        Build a cache key for one or multiple reference files.
        """

        paths = self._as_paths(reference_audio)

        return tuple(
            self._file_cache_key(path)
            for path in paths
        )

    # ========================================================
    # Reference embedding extraction
    # ========================================================

    def extract_reference_embedding(
        self,
        reference_audio: ReferenceInput,
    ) -> torch.Tensor:
        """
        Create an enrolled speaker embedding.

        A single reference:
            reference.wav

        Multiple references:
            [reference1.wav, reference2.wav, reference3.wav]

        For multiple recordings:
            1. Extract an embedding from each recording.
            2. Normalize each embedding.
            3. Average the embeddings.
            4. Normalize the resulting profile.
        """

        paths = self._as_paths(reference_audio)

        cache_key = self._reference_key(
            paths
        )

        # Reuse cache when the reference files have not changed.
        if (
            self._reference_embedding is not None
            and self._reference_cache_key == cache_key
        ):
            return self._reference_embedding

        embeddings: List[torch.Tensor] = []

        for path in paths:

            audio = self._load_audio(path)

            duration = (
                len(audio) / TARGET_SR
            )

            if duration < MIN_REFERENCE_SECONDS:
                raise ValueError(
                    f"Reference audio is too short: {path}. "
                    f"At least "
                    f"{MIN_REFERENCE_SECONDS:.1f} "
                    f"second is required."
                )

            embedding = self._encode_audio(
                audio
            )

            embeddings.append(
                embedding
            )

        if not embeddings:
            raise ValueError(
                "No valid reference embeddings were created."
            )

        # One reference.
        if len(embeddings) == 1:

            reference_embedding = embeddings[0]

        # Multiple references.
        else:

            stacked = torch.cat(
                embeddings,
                dim=0,
            )

            reference_embedding = torch.mean(
                stacked,
                dim=0,
                keepdim=True,
            )

            reference_embedding = F.normalize(
                reference_embedding,
                p=2,
                dim=-1,
            )

        self._reference_embedding = (
            reference_embedding
        )

        self._reference_cache_key = (
            cache_key
        )

        return reference_embedding

    # ========================================================
    # Score calculation
    # ========================================================

    def score_with_embedding(
        self,
        reference_embedding: torch.Tensor,
        test_audio: AudioInput,
    ) -> float:
        """
        Return the raw cosine similarity score.

        No threshold decision is applied here.

        This method is used by the calibration/evaluation pipeline.
        """

        if reference_embedding is None:
            raise ValueError(
                "Reference speaker embedding is missing."
            )

        if not isinstance(
            reference_embedding,
            torch.Tensor,
        ):
            raise ValueError(
                "Reference speaker embedding is invalid."
            )

        if reference_embedding.numel() == 0:
            raise ValueError(
                "Reference speaker embedding is empty."
            )

        # Make sure shape is [1, embedding_dim].
        reference_embedding = (
            reference_embedding
            .reshape(
                reference_embedding.shape[0],
                -1,
            )
            .to(self.device)
        )

        reference_embedding = F.normalize(
            reference_embedding,
            p=2,
            dim=-1,
        )

        test_audio_tensor = self._load_audio(
            test_audio
        )

        duration = (
            len(test_audio_tensor) / TARGET_SR
        )

        if duration < MIN_TEST_SECONDS:
            raise ValueError(
                f"Test audio is too short: "
                f"{test_audio}. "
                f"At least "
                f"{MIN_TEST_SECONDS:.1f} "
                f"second is required."
            )

        test_embedding = self._encode_audio(
            test_audio_tensor
        )

        # SpeechBrain's SpeakerRecognition class uses
        # cosine similarity for speaker verification.
        with torch.inference_mode():

            score = self.model.similarity(
                reference_embedding,
                test_embedding,
            )

        score_value = float(
            score.squeeze().item()
        )

        if not np.isfinite(score_value):
            raise ValueError(
                "ECAPA produced an invalid similarity score."
            )

        return score_value

    def score(
        self,
        reference_audio: ReferenceInput,
        test_audio: AudioInput,
    ) -> float:
        """
        Compute a raw reference-vs-test speaker similarity score.
        """

        reference_embedding = (
            self.extract_reference_embedding(
                reference_audio
            )
        )

        return self.score_with_embedding(
            reference_embedding,
            test_audio,
        )

    # ========================================================
    # Verification
    # ========================================================

    def verify_with_embedding(
        self,
        reference_embedding: torch.Tensor,
        test_audio: AudioInput,
    ) -> Dict:
        """
        Compare test audio against an existing reference embedding.
        """

        try:

            score_value = self.score_with_embedding(
                reference_embedding,
                test_audio,
            )

            same_speaker = (
                score_value >= self.threshold
            )

            return {
                "status": "success",
                "voice_match_score": score_value,
                "threshold": self.threshold,
                "same_speaker": same_speaker,
                "prediction": (
                    "same_speaker"
                    if same_speaker
                    else "different_speaker"
                ),
                "message": (
                    "Reference and test recordings "
                    "were compared successfully."
                ),
                "details": {
                    "device": self.device,
                    "threshold": self.threshold,
                    "score": score_value,
                },
            }

        except ValueError as exc:

            return {
                "status": "error",
                "voice_match_score": None,
                "threshold": self.threshold,
                "same_speaker": False,
                "prediction": None,
                "message": str(exc),
                "details": {},
            }

    def verify(
        self,
        reference_audio: ReferenceInput,
        test_audio: AudioInput,
    ) -> Dict:
        """
        Backward-compatible public API.

        Supports:

            verifier.verify(
                "reference.wav",
                "test.wav",
            )

        or:

            verifier.verify(
                [
                    "reference1.wav",
                    "reference2.wav",
                    "reference3.wav",
                ],
                "test.wav",
            )
        """

        try:

            reference_embedding = (
                self.extract_reference_embedding(
                    reference_audio
                )
            )

            return self.verify_with_embedding(
                reference_embedding,
                test_audio,
            )

        except ValueError as exc:

            return {
                "status": "error",
                "voice_match_score": None,
                "threshold": self.threshold,
                "same_speaker": False,
                "prediction": None,
                "message": str(exc),
                "details": {},
            }

    # ========================================================
    # Model / status information
    # ========================================================

    def get_info(self) -> Dict:
        """
        Return useful runtime information.
        """

        embedding_shape = None

        if (
            self._reference_embedding
            is not None
        ):
            embedding_shape = list(
                self._reference_embedding.shape
            )

        return {
            "model": self.model_source,
            "device": self.device,
            "target_sample_rate": TARGET_SR,
            "threshold": self.threshold,
            "reference_cached": (
                self._reference_embedding
                is not None
            ),
            "reference_embedding_shape":
                embedding_shape,
        }


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "ECAPA-TDNN speaker verification "
            "for VoxFusion."
        )
    )

    parser.add_argument(
        "reference_audio",
        nargs="+",
        help=(
            "One or more reference recordings."
        ),
    )

    parser.add_argument(
        "--test",
        required=True,
        help=(
            "Suspicious/test recording."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_VERIFICATION_THRESHOLD,
        help=(
            "Speaker verification threshold. "
            "Default: 0.25"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )

    args = parser.parse_args()

    verifier = SpeakerVerifier(
        threshold=args.threshold
    )

    # One reference or multiple references.
    if len(args.reference_audio) == 1:
        reference_input = args.reference_audio[0]
    else:
        reference_input = args.reference_audio

    result = verifier.verify(
        reference_input,
        args.test,
    )

    if args.json:

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        return

    print(
        "\n========== ECAPA RESULT =========="
    )
    print(
        f"Voice Match Score : "
        f"{result.get('voice_match_score')}"
    )
    print(
        f"Threshold         : "
        f"{result.get('threshold')}"
    )
    print(
        f"Same Speaker      : "
        f"{result.get('same_speaker')}"
    )
    print(
        f"Prediction        : "
        f"{result.get('prediction')}"
    )
    print(
        f"Status            : "
        f"{result.get('status')}"
    )
    print(
        f"Message           : "
        f"{result.get('message')}"
    )
    print(
        "=================================="
    )


if __name__ == "__main__":
    main()