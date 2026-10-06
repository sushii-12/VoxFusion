# VoxFusion — Voice-Cloning Family-Emergency Scam Detection

**Team:** AltF4  
**Project Type:** Academic Capstone  
**Domain:** Audio Deepfake Detection · Speaker Verification · Scam Detection

VoxFusion is a multimodal voice-cloning scam detection system designed to identify potential **family-emergency voice-cloning scams** by combining personalized speaker verification, codec-aware deepfake detection, speech transcription, conversational scam-intent analysis, and an interpretable fusion engine.

> **Scope:** VoxFusion is an **audio-file analysis prototype** for academic research and demonstration. It does not intercept live phone calls or integrate with carrier networks.

---

## Research Question

> **Does combining personalized speaker verification with codec-robust deepfake detection and conversational intent analysis improve detection of voice-cloning family-emergency scams compared to deepfake detection alone?**

The project compares:

- **Baseline:** AASIST deepfake detection alone
- **VoxFusion:** ECAPA-TDNN + AASIST + faster-whisper + scam-intent analysis + interpretable fusion

The current evaluation provides quantitative evidence for the individual components and qualitative end-to-end demonstrations. It does **not** claim statistically proven end-to-end superiority because a jointly labelled dataset covering speaker identity, deepfake status, codec conditions, scam intent, and final scam outcome is not available.

---

# At a Glance

| Component | Final Result |
|---|---|
| **AASIST V2 — Clean** | **90.00% Accuracy / 90.48% F1** |
| **AASIST V2 — GSM** | **77.50% Accuracy / 81.63% F1** |
| **AASIST GSM Diagnostic** | **74.00% Accuracy / 0.9097 ROC-AUC** |
| **ECAPA-TDNN** | **98.65% Accuracy / 1.30% EER** |
| **Scam Intent V2 — Validation** | **97.25% F1** |
| **Scam Intent V2 — Test** | **92.42% Accuracy / 96.05% F1** |
| **End-to-End** | **3 qualitative cases** |

### Key AASIST V2 improvement under GSM

```text
Accuracy:        67.50% → 77.50%
F1-score:        75.47% → 81.63%
False positives:      13 → 9
Spoof recall:        100% → 100%
```

---

# System Architecture

```text
                    RECORDED / UPLOADED AUDIO
                               |
                               v
                       AUDIO PREPROCESSING
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
        ECAPA-TDNN          AASIST        faster-whisper
      Speaker Verification Deepfake       Transcription
             |              Detection            |
             |                                   v
             |                           Scam Intent V1/V2
             |                                   |
             +-----------------+-----------------+
                               |
                               v
                    INTERPRETABLE FUSION ENGINE
                               |
                               v
                         RISK SCORE 0–100
                               |
                    +----------+----------+
                    |          |          |
                   LOW       MEDIUM      HIGH
```

## Component Roles

| Component | Question Answered | Technique |
|---|---|---|
| ECAPA-TDNN | Does the voice match the registered family member? | Speaker embeddings + cosine similarity |
| AASIST | Is the audio genuine or AI-generated/spoofed? | Audio anti-spoofing |
| faster-whisper | What is being said? | Speech-to-text |
| Scam Intent V1 | Are known scam indicators present? | Rule-based analysis |
| Scam Intent V2 | Does the transcript resemble scam language? | TF-IDF + Logistic Regression |
| Fusion Engine | How risky is the call overall? | Interpretable weighted + interaction rules |

---

# Components

## 1. ECAPA-TDNN Personalized Speaker Verification

VoxFusion uses a pretrained **ECAPA-TDNN** speaker-verification model through SpeechBrain.

A family member can provide a reference voice recording. The system extracts a speaker embedding and compares suspicious audio against that reference using cosine similarity.

Possible statuses include:

- `verified`
- `different_speaker`
- `not_verified`
- error / unavailable reference

A missing ECAPA reference **does not stop the other branches**. AASIST, Whisper, intent analysis, and fusion continue to execute.

### Threshold

The current production threshold is:

```text
0.2951
```

The latest calibration produced:

```text
EER:                    1.30%
EER-derived threshold:  0.2946
```

The production threshold remains `0.2951`; the difference from the freshly derived EER threshold is negligible.

ECAPA cosine similarity is a similarity measure, **not a probability**.

---

## 2. AASIST Deepfake Detection

VoxFusion uses the AASIST architecture for audio deepfake/spoof detection.

The project contains both the original pretrained AASIST checkpoint used for baseline evaluation and the GSM-aware fine-tuned V2 model used as the production model.

### Original baseline checkpoint

```text
aasist/models/weights/AASIST.pth
```

### Production AASIST V2 checkpoint

```text
models/AASIST_gsm_finetuned_best.pth
```

The implementation:

- accepts audio files
- converts audio to mono
- resamples to 16 kHz
- prepares the expected AASIST input length
- performs inference using the available CUDA/CPU device

AASIST is the project's primary **deepfake-only baseline**.

---

## 3. GSM / Telephone Codec Simulation

To study telephone-style degradation, VoxFusion uses **FFmpeg** to simulate GSM compression.

```text
Original WAV
    |
    v
8 kHz mono GSM codec
    |
    v
Decoded PCM WAV
    |
    v
AASIST / VoxFusion
```

This experiment demonstrated that telephone-style compression can significantly increase false alarms in deepfake detection.

### GSM score shift

| Audio | Clean Mean Score | GSM Mean Score | Change |
|---|---:|---:|---:|
| Bona fide | 0.273253 | 0.532050 | **+0.258796** |
| Spoof | 0.976989 | 0.998902 | **+0.021913** |

The increase for genuine speech was substantially larger than for spoofed speech, motivating GSM-aware fine-tuning.

---

## 4. faster-whisper Transcription

VoxFusion uses **faster-whisper** to convert suspicious audio into text.

The resulting transcript is passed to the scam-intent analysis layer.

---

## 5. Scam Intent V1

V1 is a rule-based scam-intent analyzer.

It checks for conversational indicators such as:

- emergency situations
- money requests
- urgency
- secrecy
- police/legal pressure
- suspicious payment or transfer requests
- family-emergency patterns

V1 is retained as an interpretable baseline signal.

---

## 6. Scam Intent V2

V2 is a lightweight machine-learning classifier using:

```text
TF-IDF
   +
Logistic Regression
```

It operates on transcript-level information and was developed using the **ICFD-31k** dataset.

The dataset is substantially class-imbalanced, so V2 is treated as a supporting signal rather than definitive proof of scam intent.

---

## 7. Interpretable Fusion Layer

The fusion engine combines:

```text
Speaker Verification
        +
AASIST Deepfake Evidence
        +
Scam Intent Evidence
        |
        v
   Fusion Engine
        |
        v
Risk Score + Reasons + Risk Level
```

### Current scoring semantics

- AASIST contribution: up to **35 points**
- Intent contribution: up to **30 points**
- Spoof + speaker match + elevated intent: **+12**
- Spoof + speaker mismatch + elevated intent: **+6**

### Risk levels

```text
0–29.99     LOW
30–59.99    MEDIUM
60+         HIGH
```

A speaker mismatch **alone is not treated as proof of a scam**.

A speaker match also does not prove authenticity because an AI-cloned voice may successfully match the registered family member.

The fusion score is **not a calibrated probability**. Current fusion weights and thresholds remain provisional research-prototype parameters.

---

# Web Application

VoxFusion includes a local React + FastAPI web application for demonstrating the complete pipeline.

## Frontend

The frontend uses:

- React
- Vite
- JavaScript
- CSS

The interface supports:

- family-member management
- reference voice management
- audio/sample management
- combined VoxFusion analysis
- AASIST-only baseline analysis
- AASIST-only vs combined comparison
- component-level results
- fusion risk and reasons
- system readiness/status

## Backend

The backend uses:

- FastAPI
- Python
- SQLite for local demo persistence
- local audio-file storage

Backend location:

```text
frontend/backend/app/
```

Backend URL:

```text
http://127.0.0.1:8001
```

---

# Analysis Workflow

```text
1. Register/select family member
          |
2. Provide reference voice
          |
3. Upload suspicious audio
          |
4. Select analysis mode
       |          |
       v          v
 AASIST-only   Combined
                  |
        +---------+---------+
        |         |         |
      ECAPA     AASIST    Whisper
                            |
                       Intent V1/V2
                            |
                            v
                         Fusion
                            |
                            v
                       Risk + Reasons
```

If no ECAPA reference is available, the speaker branch returns a no-reference/not-verified result while the other branches continue normally.

---

# Project Structure

```text
VoxFusion/
│
├── aasist/                         # AASIST Git submodule
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
│   │   └── data/                   # local runtime data
│   │
│   └── frontend/
│       ├── src/
│       ├── package.json
│       └── ...
│
├── scripts/
│   ├── create_ecapa_trials.py
│   ├── evaluate_ecapa.py
│   ├── evaluate_aasist_finetuned_test.py
│   └── ...
│
├── tests/
│   └── fixtures/
│       ├── person4_test.wav
│       └── person4_test_phone.wav
│
├── data/                           # ignored by Git
├── models/                         # ignored by Git
├── results/                        # ignored by Git
├── requirements.txt
├── README.md
├── .gitignore
└── .gitmodules
```

Datasets, model weights, generated results, local databases, uploaded audio, and other runtime artifacts are intentionally excluded from version control.

---

# Setup

## 1. Clone the Repository

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
.\env\Scripts\Activate.ps1
```

Install root dependencies:

```powershell
pip install -r requirements.txt
```

---

## 3. AASIST Checkpoints

The AASIST repository must be initialized and the required pretrained checkpoint must be available.

Original baseline:

```text
aasist/models/weights/AASIST.pth
```

Production V2:

```text
models/AASIST_gsm_finetuned_best.pth
```

Model weights are intentionally not committed to Git.

---

## 4. FFmpeg

FFmpeg is required for GSM codec simulation and related audio-processing workflows.

Verify:

```powershell
ffmpeg -version
```

If the command is not recognized, install FFmpeg and make sure it is available on the system PATH.

---

# Running the Web Application

The application uses two processes:

- FastAPI backend
- React + Vite frontend

## Backend

From the repository root:

```powershell
python -m uvicorn frontend/backend/app/main:app --host 127.0.0.1 --port 8001
```

Backend:

```text
http://127.0.0.1:8001
```

Health endpoint:

```text
http://127.0.0.1:8001/api/health
```

The health endpoint reports the readiness of core components including:

- AASIST
- ECAPA
- Whisper
- Scam Intent V1
- Scam Intent V2
- Fusion

---

## Frontend

Open another terminal:

```powershell
cd frontend/frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

The frontend communicates with the FastAPI backend on port `8001`.

---

# Local Runtime Data

The web application currently uses local storage for demonstration purposes.

SQLite database:

```text
frontend/backend/data/
```

Uploaded audio:

```text
frontend/backend/data/uploads/
```

These are local demo/runtime files and are intentionally ignored by Git. They may contain:

- registered family-member information
- uploaded audio
- analysis records
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

Cloning the repository therefore does not include local recordings, datasets, model weights, or generated evaluation artifacts.

---

# Evaluation

VoxFusion evaluation is divided into:

1. **Component-level quantitative evaluation**
2. **Qualitative end-to-end case studies**

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

---

# ECAPA Speaker Verification Evaluation

ECAPA-TDNN was evaluated using a speaker-disjoint procedure based on **LibriSpeech test-clean**.

The evaluation was divided into:

1. calibration speakers
2. held-out speakers

### Calibration

```text
Calibration speakers: 20
Calibration EER: 1.30%
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

The evaluation demonstrates strong speaker-discrimination performance on the selected clean read-speech evaluation setup. However, **LibriSpeech is clean read speech**. These results do **not** establish ECAPA performance under telephony compression, GSM codecs, noisy calls, or real-world family-emergency conversations.

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

These cases demonstrate that the full system can incorporate conversational evidence that AASIST alone cannot see. However, no cloned-voice positive case was included in this final qualitative set. Therefore, these cases do not establish end-to-end detection accuracy or prove that VoxFusion outperforms AASIST.

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

Fusion scores should **not** be interpreted as calibrated probabilities. The current fusion weights and thresholds are provisional research-prototype parameters.

---

# Scam Intent Evaluation

## V1 — Rule-Based Baseline

Rule-based intent analysis provides an interpretable baseline using scam-related conversational indicators.

## V2 — TF-IDF + Logistic Regression

V2 uses:

```text
TF-IDF
   +
Logistic Regression
```

The model was trained using transcript-level information from the **ICFD-31k** dataset.

The available dataset has substantial class imbalance, so V2 results are treated as a supporting signal rather than definitive proof of scam intent.

---

# Limitations

VoxFusion is an academic research prototype and has several important limitations.

### AASIST false positives

AASIST produced false-positive deepfake classifications on some genuine codec-compressed and phone-quality examples.

### Provisional fusion

The current fusion weights and risk thresholds are provisional.

They have not been calibrated against a sufficiently large, representative end-to-end dataset.

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

The project investigates whether multiple independent signals can provide more useful contextual evidence for voice-cloning family-emergency scam detection than relying on deepfake detection alone.

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

---
