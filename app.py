import argparse
import json
from pathlib import Path

import soundfile as sf

from src.speaker_verification import SpeakerVerifier
from src.deepfake_detection import AASISTDetector
from src.transcription import WhisperTranscriber
from src.scam_intent import ScamIntentAnalyzer
from src.fusion import FusionEngine


def run_voxfusion(audio_path, reference_audio=None):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if reference_audio is not None:
        reference_audio = Path(reference_audio)

        if not reference_audio.is_file():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio}"
            )

    print("\n========== VOXFUSION ==========")
    print(f"Input audio: {audio_path}")

    # Initialize the modules.
    # Models are loaded once per application run.
    print("\nInitializing models...")

    speaker_verifier = SpeakerVerifier()
    deepfake_detector = AASISTDetector()
    transcriber = WhisperTranscriber()

    intent_analyzer = ScamIntentAnalyzer()
    fusion_engine = FusionEngine()

    # --------------------------------------------------
    # 1. ECAPA-TDNN speaker verification
    # --------------------------------------------------
    print("\n[1/3] Running speaker verification...")

    speaker_result = None

    if reference_audio is not None:
        speaker_result = speaker_verifier.verify(
            str(reference_audio),
            str(audio_path),
        )
    else:
        print("No reference audio supplied.")
        print("Speaker verification: NOT VERIFIED")

    # --------------------------------------------------
    # 2. AASIST deepfake detection
    # --------------------------------------------------
    print("\n[2/3] Running AASIST...")

    try:
        deepfake_result = deepfake_detector.predict(
            str(audio_path)
        )
    except sf.LibsndfileError as exc:
        raise ValueError(
            f"Unable to read audio file for AASIST: {audio_path}. "
            "Please provide a valid WAV/audio file."
        ) from exc

    # --------------------------------------------------
    # 3. Whisper transcription + scam-intent analysis
    # --------------------------------------------------
    print("\n[3/3] Running Whisper transcription...")

    transcription_result = transcriber.transcribe(
        str(audio_path)
    )

    transcript = transcription_result["transcript"]

    print("\nAnalyzing scam intent...")

    intent_result = intent_analyzer.analyze(transcript)

    # --------------------------------------------------
    # 4. Fusion
    # --------------------------------------------------
    print("\nCombining results...")

    fusion_result = fusion_engine.analyze(
        speaker_result=speaker_result,
        deepfake_result=deepfake_result,
        intent_result=intent_result,
    )

    # --------------------------------------------------
    # 5. Combined output
    # --------------------------------------------------
    result = {
        "audio_file": str(audio_path),
        "speaker_verification": (
            speaker_result
            if speaker_result is not None
            else {
                "status": "not_verified",
                "reason": "No reference audio supplied",
            }
        ),
        "deepfake_detection": deepfake_result,
        "transcription": transcription_result,
        "scam_intent": intent_result,
        "fusion": fusion_result,
    }

    print("\n========== VOXFUSION RESULTS ==========")

    print("\nSpeaker verification:")
    print(json.dumps(result["speaker_verification"], indent=2))

    print("\nDeepfake detection:")
    print(json.dumps(deepfake_result, indent=2))

    print("\nTranscription:")
    print(transcript)

    print("\nScam intent:")
    print(json.dumps(intent_result, indent=2))

    print("\nOVERALL RISK:")
    print(f"Score: {fusion_result['risk_score']}/100")
    print(f"Level: {fusion_result['risk_level']}")

    print("\nReasons:")
    for reason in fusion_result["reasons"]:
        print(f"- {reason}")

    print("\n========================================")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="VoxFusion voice-clone scam detection"
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the suspicious call audio",
    )

    parser.add_argument(
        "--reference",
        default=None,
        help="Optional registered family-member voice recording",
    )

    args = parser.parse_args()

    try:
        run_voxfusion(
            audio_path=args.audio,
            reference_audio=args.reference,
        )
    except ValueError as exc:
        print(f"\nError: {exc}")


if __name__ == "__main__":
    main()