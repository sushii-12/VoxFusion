# VoxFusion - Voice-Cloning Family-Emergency Scam Detection

**Team:** AltF4  
**Project Type:** Academic Capstone

VoxFusion is a multimodal voice-cloning scam detection system designed to identify potential **family-emergency voice-cloning scams** by combining personalized speaker verification, deepfake detection, and conversational scam-intent analysis.

The system is designed as an **audio-file analysis prototype**. It does not perform live-call interception.

---

## Research Question

> **Does combining personalized speaker verification with codec-robust deepfake detection and conversational intent analysis improve detection of voice-cloning family-emergency scams compared to deepfake detection alone?**

The project evaluates this question by comparing:

- **Baseline:** AASIST deepfake detection alone
- **VoxFusion:** ECAPA-TDNN + AASIST + faster-whisper + scam-intent analysis + interpretable fusion

The comparison focuses on whether multimodal evidence provides useful contextual information beyond deepfake detection alone. The current project does **not** claim statistically proven end-to-end superiority because a combined ground-truth dataset for the complete VoxFusion pipeline is not available.

---

# System Architecture

VoxFusion processes a suspicious audio recording through multiple independent signals.

```text
                    RECORDED / UPLOADED AUDIO
                              |
                       AUDIO PREPROCESSING
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        ECAPA-TDNN          AASIST        faster-whisper
       Speaker Match      Deepfake        Transcription
              |            Detection            |
              |               |                  v
              |               |          Scam Intent Analysis
              |               |             V1 + V2
              |               |                  |
              +---------------+------------------+
                              |
                       INTERPRETABLE
                       FUSION ENGINE
                              |
                         RISK SCORE
                              |
                    +---------+---------+
                    |         |         |
                   LOW     MEDIUM     HIGH
```

## Components

### 1. ECAPA-TDNN Personalized Speaker Verification

VoxFusion uses a pretrained **ECAPA-TDNN** speaker-verification model through SpeechBrain.

A family member can provide a reference voice recording. The system extracts a speaker embedding from the reference and compares it with the suspicious audio using cosine similarity.

The system returns a speaker-verification status such as:

- `verified`
- `different_speaker`
- `not_verified`
- error / unavailable reference

A missing ECAPA reference **does not prevent the other branches from running**. AASIST, Whisper, intent analysis, and fusion continue to execute even when no suitable reference voice is available.

The current verification threshold is **0.2951**, derived from a speaker-disjoint LibriSpeech calibration procedure. The threshold is provisional for the project and does not establish telephony-domain performance.

---

### 2. AASIST Deepfake Detection

VoxFusion uses the pretrained **AASIST** architecture and checkpoint for audio deepfake detection.

The model analyzes the audio and produces a deepfake/bona-fide prediction and associated scores.

The implementation:

- accepts audio files
- converts audio to mono
- resamples to 16 kHz
- prepares the expected AASIST input length
- performs inference using the available model checkpoint

AASIST is also used as the project's primary **baseline system** for comparison against the complete VoxFusion pipeline.

---

### 3. GSM / Telephone Codec Simulation

To study the effect of telephone-style audio degradation, VoxFusion uses **FFmpeg** to simulate GSM/telephone codec processing.

The general evaluation path is:

```text
Original WAV
    |
    v
GSM Codec Compression
    |
    v
Compressed Audio
    |
    v
VoxFusion / AASIST
```

This allows the project to examine how codec compression and phone-quality audio affect deepfake detection and the other system components.

---

### 4. faster-whisper Transcription

The system uses **faster-whisper** to convert the suspicious audio into text.

The resulting transcript is passed to the scam-intent analysis layer.

The transcription branch operates independently of ECAPA speaker verification.

---

### 5. Scam Intent V1

V1 is a **rule-based scam-intent analyzer**.

It looks for conversational indicators associated with family-emergency scams, including concepts such as:

- emergency situations
- requests for money
- urgency
- secrecy
- suspicious payment requests

The V1 system provides an interpretable intent score and supporting categories.

---

### 6. Scam Intent V2

V2 is a lightweight machine-learning intent classifier based on:

- **TF-IDF**
- **Logistic Regression**

The model operates on transcript-level information and is intended to provide a second scam-intent signal alongside the rule-based V1 analyzer.

The V2 model was developed using the ICFD-31k dataset. Its evaluation is treated cautiously because the available dataset has significant class imbalance and limitations for representing real-world family-emergency scam conversations.

---

### 7. Interpretable Fusion Layer

The fusion engine combines the available signals:

```text
Speaker Verification
        +
AASIST Deepfake Score
        +
Scam Intent Evidence
        |
        v
   Fusion Engine
        |
        v
Risk Score + Reasons + Risk Level
```

The current fusion output provides:

- overall risk score
- `LOW`
- `MEDIUM`
- `HIGH`
- contributing reasons
- component-level evidence

The fusion score is **not a calibrated probability**. The current fusion weights and thresholds are provisional and are intended for the research prototype.

A speaker mismatch by itself is not treated as proof of fraud.

---

# Web Application

VoxFusion now includes a local web application for demonstrating and evaluating the complete pipeline.

```text
frontend/
├── backend/
└── frontend/
```

The directory names are intentionally retained as they currently exist in the project.

## Frontend

The frontend is implemented using:

- **React**
- **Vite**
- JavaScript
- CSS

The interface provides workflows for:

- family-member management
- reference voice management
- audio/sample management
- running analyses
- selecting a target family member
- running the combined VoxFusion pipeline
- running the AASIST-only baseline
- comparing AASIST-only and combined results
- viewing component-level results
- viewing fusion risk scores and reasons
- viewing system readiness/status

---

## Backend

The backend is implemented using:

- **FastAPI**
- Python
- SQLite for local demo persistence
- local audio-file storage

The backend exposes API endpoints for:

- application health/readiness
- family-member management
- reference/sample management
- audio analysis
- AASIST-only vs combined comparison
- analysis-result retrieval

The main backend application is located under:

```text
frontend/backend/app/
```

The backend runs on:

```text
http://127.0.0.1:8001
```

---

## Analysis Workflow

A typical web-application workflow is:

```text
1. Register/select a family member
             |
2. Provide reference voice
             |
3. Upload suspicious audio
             |
4. Select analysis mode
       |               |
       v               v
 AASIST-only        Combined
                       |
          +------------+------------+
          |            |            |
       ECAPA         AASIST      Whisper
                                    |
                              Intent V1/V2
                                    |
                                    v
                                 Fusion
                                    |
                                    v
                              Risk + Reasons
```

If no ECAPA reference is available, the speaker-verification branch returns a `not_verified`/no-reference result while the remaining analysis branches continue normally.

---

# Project Structure

```text
VoxFusion/
│
├── aasist/
│   └── AASIST repository (Git submodule)
│
├── src/
│   ├── deepfake_detection.py
│   ├── speaker_verification.py
│   ├── transcription.py
│   ├── scam_intent.py
│   └── fusion.py
│
├── frontend/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   └── services/
│   │   └── data/
│   │       ├── voice_analysis.db
│   │       └── uploads/
│   │
│   └── frontend/
│       ├── src/
│       ├── package.json
│       └── ...
│
├── scripts/
│   ├── create_ecapa_trials.py
│   ├── evaluate_ecapa.py
│   └── validate_ecapa_threshold.py
│
├── tests/
│   └── fixtures/
│       ├── person4_test.wav
│       └── person4_test_phone.wav
│
├── data/                 # ignored by Git
├── models/               # ignored by Git
├── results/              # ignored by Git
│
├── requirements.txt
├── README.md
├── .gitignore
└── .gitmodules
```

Datasets, downloaded model weights, generated evaluation results, local databases, uploaded audio, and other runtime artifacts are intentionally excluded from version control.

---

# Setup

## 1. Clone the Repository

Clone VoxFusion together with the AASIST submodule:

```powershell
git clone --recurse-submodules https://github.com/sushii-12/VoxFusion.git
cd VoxFusion
```

If the repository was already cloned without submodules:

```powershell
git submodule update --init --recursive
```

---

## 2. Python Environment

VoxFusion uses **Python 3.11**.

Create the environment:

```powershell
python -m venv env
```

Activate it:

```powershell
.\\env\\Scripts\\Activate.ps1
```

Install the root Python dependencies:

```powershell
pip install -r requirements.txt
```

---

## 3. AASIST Checkpoint

The AASIST repository must be initialized and the required pretrained checkpoint must be available.

The project currently uses:

```text
aasist/models/weights/AASIST.pth
```

The checkpoint is intentionally not committed to Git.

---

## 4. FFmpeg

FFmpeg is required for the telephone/GSM codec simulation and related audio-processing workflows.

Verify that FFmpeg is available:

```powershell
ffmpeg -version
```

If the command is not recognized, install FFmpeg and make sure it is available on the system PATH.

---

# Running the Web Application

The application consists of two processes:

- FastAPI backend
- React/Vite frontend

Start them separately.

## Backend

From the repository root:

```powershell
python -m uvicorn frontend.backend.app.main:app --host 127.0.0.1 --port 8001
```

The backend runs on:

```text
http://127.0.0.1:8001
```

The FastAPI application also exposes a health/readiness endpoint used to report the availability of core components such as AASIST, ECAPA, Whisper, intent analysis, and fusion.

---

## Frontend

Open another terminal and activate the environment if necessary.

Then:

```powershell
cd frontend/frontend
npm install
npm run dev
```

The Vite development server runs on:

```text
http://localhost:5173
```

The frontend communicates with the FastAPI backend running on port `8001`.

---

# Local Runtime Data

The web application currently uses local storage for demonstration purposes.

The local SQLite database is stored under:

```text
frontend/backend/data/
```

Uploaded audio is stored under:

```text
frontend/backend/data/uploads/
```

These files represent **local demo/runtime data**.

They are intentionally ignored by Git because they can contain:

- locally registered family-member information
- uploaded audio
- generated analysis records
- runtime state

The following categories are also intentionally ignored:

```text
data/
models/
results/
frontend/backend/data/*.db
frontend/backend/data/uploads/
frontend/frontend/node_modules/
frontend/frontend/dist/
frontend/frontend/.vite/
```

Therefore, cloning the repository does not include local runtime recordings, datasets, model weights, or generated evaluation artifacts.

---

# Evaluation

VoxFusion evaluation is divided into component-level evaluation and qualitative end-to-end case studies.

The reported AASIST and ECAPA results below were obtained from prepared local evaluation subsets and should not be interpreted as full-corpus or production performance.

---

## AASIST Evaluation

AASIST was evaluated on prepared local subsets of ASVspoof data.

### ASVspoof 2019 LA

**Local prepared subset: 200 samples**

| Metric | Result |
|---|---:|
| Accuracy | 0.98 |
| Precision | 1.00 |
| Recall | 0.96 |
| F1 | 0.9796 |
| False Negative Rate | 0.04 |

### ASVspoof 2021 LA

**Local prepared subset: 200 samples**

| Metric | Result |
|---|---:|
| Accuracy | 0.845 |
| Precision | 0.7717 |
| Recall | 0.98 |
| F1 | 0.8634 |
| False Negative Rate | 0.02 |

These are **prepared local subsets**, not complete ASVspoof 2019 LA or ASVspoof 2021 LA datasets.

The difference between the two subsets also demonstrates that deepfake detection performance can change under different data and channel conditions. In particular, false positives were observed in the 2021 subset.

---

# ECAPA Speaker Verification Evaluation

ECAPA-TDNN was evaluated using a speaker-disjoint procedure based on **LibriSpeech test-clean**.

The evaluation was divided into:

1. calibration speakers
2. held-out speakers

The calibration split was used to determine the operating threshold.

### Calibration

```text
Calibration speakers: 20
Calibration EER: 1.27%
Selected threshold: 0.2951
```

### Held-Out Speakers

| Metric | Result |
|---|---:|
| FAR | 4.4% |
| FRR | 0.8% |
| Accuracy | 97.4% |
| Precision | 98.90% |
| Recall | 99.2% |
| F1 | 99.05% |

This evaluation demonstrates strong speaker-discrimination performance on the selected clean read-speech evaluation setup.

However, **LibriSpeech is clean read speech**. These results do **not** establish ECAPA performance under telephony compression, GSM codecs, noisy calls, or real-world family-emergency conversations.

---

# Case-Study Findings

The project also includes qualitative case studies comparing:

```text
AASIST-only baseline
        vs
VoxFusion combined system
```

Observed cases include:

### Genuine speech under GSM/telephone-style processing

In some genuine speech examples, AASIST produced a very high deepfake score after codec processing.

When additional signals were available, ECAPA speaker verification and conversational intent evidence provided contextual information that reduced the resulting overall risk severity from the AASIST-only HIGH classification.

### Genuine phone-quality speech with a matching speaker

A phone-quality genuine-speaker example produced a strong AASIST deepfake score, while ECAPA provided strong speaker-match evidence.

The combined system therefore produced a lower risk level than the AASIST-only baseline.

### Known spoof example

For a known synthetic spoof, AASIST produced a high deepfake score and ECAPA provided different-speaker evidence.

Multiple signals therefore agreed with the suspicious interpretation, and the combined system retained a HIGH risk classification.

These cases demonstrate that multimodal fusion can provide **contextual evidence beyond AASIST alone**.

They do **not** establish statistically proven end-to-end superiority, and the project does not claim that VoxFusion currently has higher accuracy than AASIST.

---

# Fusion Interpretation

The current fusion engine combines:

- AASIST deepfake evidence
- ECAPA speaker-verification evidence when a reference is available
- scam-intent evidence from V1/V2

The output includes an interpretable risk score and reasons.

Example reasoning can include:

```text
- Deepfake evidence detected
- Speaker verified against registered reference
- Different speaker detected
- Scam-intent indicators detected
- No suitable speaker reference available
```

Fusion scores should **not** be interpreted as calibrated probabilities.

The current fusion weights and thresholds are provisional research-prototype parameters.

---

# Scam Intent Evaluation Notes

The project contains two intent-analysis approaches.

## V1

Rule-based intent analysis provides an interpretable baseline using scam-related conversational indicators.

## V2

V2 uses:

```text
TF-IDF
   +
Logistic Regression
```

The model was trained using transcript-level information from ICFD-31k.

The available dataset has substantial class imbalance, and the observed evaluation results show weaker non-scam recall than the headline overall metrics might suggest.

Therefore, V2 results are treated as an additional signal rather than definitive evidence of scam intent.

---

# Limitations

VoxFusion is an academic research prototype and has several important limitations.

### AASIST false positives

AASIST produced false-positive deepfake classifications on some genuine codec-compressed and phone-quality examples.

This demonstrates the importance of evaluating deepfake detectors under channel and codec shifts.

### Provisional fusion

The current fusion weights and risk thresholds are provisional.

They have not been calibrated against a sufficiently large, representative end-to-end scam dataset.

### ECAPA scores are not probabilities

ECAPA cosine similarity is a speaker-similarity measure.

It should not be interpreted directly as the probability that two recordings belong to the same person.

### Speaker mismatch is not proof of fraud

A different-speaker result can occur for legitimate reasons, including:

- a family member other than the selected reference
- a poor or unsuitable reference recording
- recording/channel differences
- another legitimate speaker

Therefore, speaker mismatch alone is not sufficient evidence of fraud.

### Intent-model limitations

The intent classifier is affected by the limitations and class imbalance of its training data.

Real-world family-emergency conversations can also differ substantially from benchmark datasets.

### No combined ground-truth dataset

The project currently does not have a sufficiently large, jointly labeled dataset containing:

- speaker identity
- deepfake status
- telephone/codec conditions
- scam intent
- end-to-end scam outcome

Therefore, a statistically rigorous end-to-end VoxFusion accuracy claim cannot currently be made.

### Audio-file analysis only

VoxFusion currently analyzes uploaded audio files.

It is **not a live-call interception system**, does not integrate directly with a phone network, and does not claim real-time carrier-level call monitoring.

### Prototype status

The web application is a functional academic demonstration and research prototype.

It should not be considered a production fraud-detection system.

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | React + Vite |
| Database | SQLite (local demo) |
| Speaker Verification | SpeechBrain ECAPA-TDNN |
| Deepfake Detection | AASIST |
| Transcription | faster-whisper |
| Intent V1 | Rule-based analysis |
| Intent V2 | TF-IDF + Logistic Regression |
| Fusion | Custom interpretable fusion engine |
| Audio Processing | SoundFile, SciPy |
| Codec Simulation | FFmpeg / GSM |
| ML Framework | PyTorch |
| Numerical Processing | NumPy |
| Version Control | Git / GitHub |

---

# Academic Scope

VoxFusion is developed as an **academic capstone project** by Team AltF4.

The project focuses on investigating whether combining multiple independent signals can provide more useful contextual evidence for voice-cloning family-emergency scam detection than relying on deepfake detection alone.

The current implementation prioritizes:

- reproducible component evaluation
- interpretable outputs
- explicit baseline comparison
- handling missing speaker references
- codec-affected audio analysis
- honest reporting of limitations

The system is intended to support further research and experimentation rather than serve as a production fraud-detection service.

---

# Team AltF4

**VoxFusion — Voice-Cloning Family-Emergency Scam Detection**

Academic Capstone Project

The project combines speaker verification, audio deepfake detection, speech transcription, scam-intent analysis, and interpretable multimodal fusion into a single research prototype.
