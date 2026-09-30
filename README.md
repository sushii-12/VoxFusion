# VoxFusion - Voice-Cloning Scam Detection

**Team:** AltF4

### VoxFusion - Voice-Cloning Scam Detection system designed to identify potential family-emergency scams involving AI-generated or cloned voices. It combines speaker verification, audio deepfake detection, speech transcription, and scam-intent analysis to produce an interpretable risk assessment.

> **Project status:** The end-to-end V1 pipeline is integrated and executable. Model evaluation, score calibration, and validation on suitable datasets are still in progress.

## Research Question

Does combining personalized speaker verification with codec-robust deepfake detection and conversational intent analysis improve detection of voice-cloning family-emergency scams compared to deepfake detection alone?

## System Architecture

VoxFusion processes an uploaded audio recording through three analysis branches.

1. **Speaker verification â€” ECAPA-TDNN**
   Compares the suspicious recording against a registered family member's reference voice and produces a speaker similarity result.

2. **Deepfake detection â€” AASIST**
   Analyzes the audio for characteristics associated with bona fide or spoofed speech.

3. **Speech transcription â€” Whisper**
   Transcribes the recording using `faster-whisper`. A rule-based intent analyzer then looks for indicators associated with emergencies, financial requests, urgency, and secrecy.

The outputs are passed to an interpretable fusion layer, which produces an overall risk score, risk category, and explanation.

```text
                         Uploaded Audio
                               |
                    Audio Preprocessing
                               |
             +-----------------+-----------------+
             |                 |                 |
         ECAPA-TDNN          AASIST            Whisper
       Speaker Match     Deepfake Score     Transcription
             |                 |                 |
             |                 |          Scam-Intent
             |                 |            Analysis
             +-----------------+-----------------+
                               |
                         Fusion Engine
                               |
                       Overall Risk Score
                               |
                       LOW / MEDIUM / HIGH
```

## Technology Stack

| Component              | Technology              |
| ---------------------- | ----------------------- |
| Programming language   | Python                  |
| Speaker verification   | SpeechBrain, ECAPA-TDNN |
| Deepfake detection     | AASIST                  |
| Speech transcription   | faster-whisper          |
| Audio processing       | SoundFile, SciPy        |
| Audio codec simulation | FFmpeg                  |
| Numerical processing   | NumPy                   |
| Deep learning          | PyTorch                 |
| Application pipeline   | Python                  |

## Key Features

* Multi-signal analysis of suspicious audio.
* Speaker comparison against a reference recording.
* Deepfake/spoof analysis using AASIST.
* Speech transcription and rule-based scam-intent analysis.
* Interpretable fusion score and risk category.
* Component-level outputs and reasons.
* GSM codec simulation for investigating telephone-style audio degradation.


## Project Structure

```text
VoxFusion/
|-- aasist/                      # AASIST source repository (Git submodule)
|-- src/
|   |-- deepfake_detection.py    # AASIST inference wrapper
|   |-- speaker_verification.py  # ECAPA-TDNN speaker verification
|   |-- transcription.py         # faster-whisper transcription
|   |-- scam_intent.py           # Rule-based scam-intent analysis
|   |-- fusion.py                # Risk fusion and explanations
|-- data/                        # Local audio data (not committed)
|-- results/                     # Local outputs (not committed)
|-- app.py                       # Pipeline entry point
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .gitmodules
```

## Setup

### 1. Clone the repository

Clone VoxFusion with its AASIST submodule:

```powershell
git clone --recurse-submodules https://github.com/sushii-12/VoxFusion.git
cd VoxFusion
```

If you already cloned the repository without its submodules:

```powershell
git submodule update --init --recursive
```

### 2. Create a Python environment

Python 3.11 is the version used during development.

```powershell
py -3.11 -m venv env
.\env\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, use the appropriate execution-policy setting for your machine or invoke the environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The AASIST repository may require additional packages. Check its documentation and dependency instructions if an import fails.

### 4. Model checkpoints

VoxFusion uses pretrained model checkpoints; it does not train ECAPA-TDNN or AASIST from scratch during normal inference.

The AASIST wrapper expects the checkpoint at:

```text
aasist/models/weights/AASIST.pth
```

Obtain the checkpoint from the official AASIST project and place it at the expected path. Model weights are excluded from this repository, so cloning VoxFusion does not necessarily download them.

The ECAPA-TDNN pretrained model is provided through SpeechBrain and may be downloaded when the model is first initialized.

The Whisper model used by the transcription module may also require a model download on first use.

### 5. FFmpeg

FFmpeg is used for audio codec conversion and GSM simulation. Install FFmpeg and ensure that `ffmpeg` is available on your system PATH.

Check the installation:

```powershell
ffmpeg -version
```

## Running VoxFusion

Run commands from the project root:

```powershell
python app.py
```

To provide a reference recording for speaker verification, use:

```powershell
python app.py --reference "path\to\reference.wav"
```

The reference recording should contain a clear sample of the family member's voice. Use the command-line options supported by the current `app.py` implementation if your local version differs.

The pipeline runs speaker verification, deepfake analysis, transcription, intent analysis, and fusion. If no reference recording is provided, speaker verification is reported as unavailable; the other analysis branches still run.

## Current Scam-Intent Analysis

The current V1 intent analyzer is rule-based and English-focused. It checks for indicators in categories such as:

* Emergency
* Money or financial transfer
* Urgency
* Secrecy

Its score and category are heuristic indicators, not a calibrated probability that a call is a scam. Rule-based matching can miss paraphrases and may flag benign conversations.

## Evaluation Plan

The main research comparison is:

* **Baseline:** AASIST deepfake detection alone on codec-processed audio.
* **Full system:** ECAPA-TDNN speaker verification, AASIST deepfake detection, and transcription-based intent analysis combined through the fusion layer.

Planned evaluation measures include:

* Accuracy, precision, recall, and F1-score
* False negative rate and confusion matrix
* Speaker verification measures such as FAR, FRR, and EER, where suitable labeled speaker pairs are available

ASVspoof datasets can support bona fide-versus-spoof evaluation. They do not, by themselves, establish whether a conversation is a family-emergency scam. Speaker verification and scam-intent evaluation require appropriate data and labels for those tasks.

No performance results are claimed here until the relevant experiments have been run and documented.

## Current Limitations

* AASIST has produced a false positive on a benign test recording. Its performance must be evaluated on appropriate datasets before drawing conclusions.
* The current fusion weights and risk thresholds are provisional and have not been calibrated.
* AASIST's output score should not be interpreted as a calibrated real-world probability of a deepfake.
* ECAPA similarity scores require threshold selection and evaluation; they are not probabilities.
* A voice mismatch alone does not establish that a call is fraudulent.
* The current scam-intent analyzer is rule-based and English-focused.
* Performance under GSM or other telephone codecs must be measured experimentally.
* The system is currently an audio-file analysis pipeline, not a live-call interception system.

## Team Handoff

Before running experiments, confirm that the required model checkpoints and datasets are available locally.

Do not commit virtual environments, private recordings, datasets, generated results, or large model artifacts to Git. Follow the dataset's license and access conditions when downloading or sharing data.

## Acknowledgements

* [AASIST â€” Official CLovaAI repository](https://github.com/clovaai/aasist)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

VoxFusion is an academic capstone project developed by Team AltF4.


